"""
Text encoder for temporal descriptions
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Union


class TextEncoder(nn.Module):
    """Text encoder using pretrained language model"""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 512,
        freeze: bool = True,
        max_length: int = 128,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.model.config.hidden_size
        self.projection = nn.Linear(hidden_size, embedding_dim)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.set_grad_enabled(self.training):
            outputs = self.model(**encoded)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        embedding = self.projection(cls_embedding)
        return embedding


class TextEncoderCache:
    """Cache for text encodings"""
    
    def __init__(self, max_cache_size: int = 1000):
        self.cache = {}
        self.max_cache_size = max_cache_size
    
    def get(self, text: str, encoder: TextEncoder) -> torch.Tensor:
        if text not in self.cache:
            with torch.no_grad():
                encoding = encoder(text)
            
            if len(self.cache) >= self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            
            self.cache[text] = encoding
        
        return self.cache[text]
    
    def clear(self):
        self.cache.clear()
