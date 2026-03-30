"""
MinAtar-specific policies (10x10x4 input)
"""
import torch
import torch.nn as nn
import numpy as np


class MinAtarCNN(nn.Module):
    """Simple CNN for MinAtar (10x10x4)"""
    
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        
        # MinAtar-specific CNN (10x10x4 -> features)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),  # 10x10x4 -> 10x10x16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10x10 -> 5x5
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 5x5x16 -> 5x5x32
            nn.ReLU(),
            nn.Flatten(),  # 5x5x32 = 800
        )
        
        # Shared features
        self.features = nn.Sequential(
            nn.Linear(800, 128),
            nn.ReLU(),
        )
        
        # Policy and value heads
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)
        
        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"MinAtar CNN created with {total_params:,} parameters")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def get_value(self, x):
        h = self.cnn(x)
        h = self.features(h)
        return self.critic(h)
    
    def get_action_and_value(self, x, action=None):
        h = self.cnn(x)
        h = self.features(h)
        
        logits = self.actor(h)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), self.critic(h)


class MinAtarHybridCNN(nn.Module):
    """Multi-scale CNN for MinAtar"""
    
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        
        # Multi-scale features
        self.scale1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10->5
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10->5
        )
        
        # Combine scales
        self.combine = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 5x5x32
            nn.ReLU(),
            nn.Flatten(),  # 800
        )
        
        # Shared features
        self.features = nn.Sequential(
            nn.Linear(800, 128),
            nn.ReLU(),
        )
        
        # Heads
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)
        
        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"MinAtar Hybrid CNN created with {total_params:,} parameters")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def get_value(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        h = self.combine(torch.cat([s1, s2], dim=1))
        h = self.features(h)
        return self.critic(h)
    
    def get_action_and_value(self, x, action=None):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        h = self.combine(torch.cat([s1, s2], dim=1))
        h = self.features(h)
        
        logits = self.actor(h)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), self.critic(h)


class MinAtarGatedFusion(nn.Module):
    """Novel gated vision-text fusion for MinAtar"""
    
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        
        # Vision encoder (lightweight CNN)
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10->5
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # 5x5x32 = 800
            nn.Linear(800, 128),
            nn.ReLU(),
        )
        
        # Text encoder (frozen pretrained)
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.text_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Freeze text model
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # Project text to same dim as vision
        self.text_proj = nn.Linear(384, 128)
        
        # NOVEL: Gating mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [vision_weight, text_weight]
            nn.Softmax(dim=-1)
        )
        
        # Fused features
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Heads
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)
        
        # Statistics
        self.vision_gates = []
        self.text_gates = []
        
        self._init_weights()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"MinAtar Gated Fusion: {trainable:,} trainable / {total:,} total params")
    
    def _init_weights(self):
        for m in [self.vision_cnn, self.text_proj, self.gate_net, self.fusion, self.actor, self.critic]:
            for layer in m.modules() if hasattr(m, 'modules') else [m]:
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def encode_text(self, texts):
        """Encode text instructions (cached per batch)"""
        with torch.no_grad():
            tokens = self.tokenizer(texts, padding=True, truncation=True, 
                                   return_tensors='pt', max_length=32)
            tokens = {k: v.to(next(self.parameters()).device) for k, v in tokens.items()}
            outputs = self.text_model(**tokens)
            text_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        return self.text_proj(text_emb)
    
    def get_value(self, x, texts=None):
        vision_features = self.vision_cnn(x)
        
        if texts is not None:
            text_features = self.encode_text(texts)
            combined = torch.cat([vision_features, text_features], dim=-1)
            gate_weights = self.gate_net(combined)
            fused = vision_features * gate_weights[:, 0:1] + text_features * gate_weights[:, 1:2]
        else:
            fused = vision_features
        
        h = self.fusion(fused)
        return self.critic(h)
    
    def get_action_and_value(self, x, texts=None, action=None):
        vision_features = self.vision_cnn(x)
        
        if texts is not None:
            text_features = self.encode_text(texts)
            combined = torch.cat([vision_features, text_features], dim=-1)
            gate_weights = self.gate_net(combined)
            
            # Track statistics
            self.vision_gates.append(gate_weights[:, 0].mean().item())
            self.text_gates.append(gate_weights[:, 1].mean().item())
            
            fused = vision_features * gate_weights[:, 0:1] + text_features * gate_weights[:, 1:2]
        else:
            fused = vision_features
        
        h = self.fusion(fused)
        
        logits = self.actor(h)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), self.critic(h)
    
    def get_gate_statistics(self):
        """Get gating statistics"""
        if len(self.vision_gates) == 0:
            return {'vision_weight': 0.5, 'text_weight': 0.5}
        
        return {
            'vision_weight': np.mean(self.vision_gates[-100:]),
            'text_weight': np.mean(self.text_gates[-100:])
        }
