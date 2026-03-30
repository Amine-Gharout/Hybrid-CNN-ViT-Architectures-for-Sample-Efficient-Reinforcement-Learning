"""
Text history buffer for temporal descriptions
"""
import numpy as np
from typing import List, Optional, Dict
from collections import deque


class HistoryBuffer:
    """Buffer to store and generate textual summaries of recent experience"""
    
    def __init__(
        self,
        history_length: int = 5,
        action_names: Optional[List[str]] = None,
        max_text_length: int = 512,
        template_style: str = "detailed",
    ):
        self.history_length = history_length
        self.action_names = action_names
        self.max_text_length = max_text_length
        self.template_style = template_style
        
        self.buffer = deque(maxlen=history_length)
        self.total_reward = 0.0
        self.step_count = 0
    
    def add(self, action: int, reward: float, done: bool, info: Dict = None):
        self.buffer.append({
            'action': action,
            'reward': reward,
            'done': done,
            'step': self.step_count,
            'info': info or {}
        })
        
        self.total_reward += reward
        self.step_count += 1
    
    def reset(self):
        self.buffer.clear()
        self.total_reward = 0.0
        self.step_count = 0
    
    def to_text(self) -> str:
        if len(self.buffer) == 0:
            return "Episode start. No previous actions."
        
        if self.template_style == "detailed":
            text = self._generate_detailed_text()
        elif self.template_style == "compact":
            text = self._generate_compact_text()
        elif self.template_style == "narrative":
            text = self._generate_narrative_text()
        else:
            text = self._generate_detailed_text()
        
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        
        return text
    
    def _generate_detailed_text(self) -> str:
        parts = [f"Recent history (last {len(self.buffer)} steps):"]
        
        for i, transition in enumerate(self.buffer):
            action_str = self._get_action_name(transition['action'])
            reward_str = self._format_reward(transition['reward'])
            
            step_desc = f"Step {i+1}: Action={action_str}, Reward={reward_str}"
            
            if transition['done']:
                step_desc += " (Episode ended)"
            
            parts.append(step_desc)
        
        parts.append(f"Total reward so far: {self.total_reward:.2f}")
        return ". ".join(parts) + "."
    
    def _generate_compact_text(self) -> str:
        transitions = []
        
        for t in self.buffer:
            action = self._get_action_name(t['action'])
            reward = self._format_reward(t['reward'])
            transitions.append(f"({action},{reward})")
        
        return "History: " + " -> ".join(transitions)
    
    def _generate_narrative_text(self) -> str:
        if len(self.buffer) == 0:
            return "The agent has just started."
        
        recent_rewards = [t['reward'] for t in self.buffer]
        avg_reward = np.mean(recent_rewards)
        
        narrative = []
        
        if avg_reward > 0.5:
            narrative.append("The agent is performing well recently.")
        elif avg_reward < -0.5:
            narrative.append("The agent has struggled in recent steps.")
        else:
            narrative.append("The agent's recent performance is mixed.")
        
        recent_actions = [self._get_action_name(t['action']) for t in list(self.buffer)[-3:]]
        narrative.append(f"Recent actions: {', '.join(recent_actions)}.")
        narrative.append(f"Cumulative reward: {self.total_reward:.1f}.")
        
        return " ".join(narrative)
    
    def _get_action_name(self, action: int) -> str:
        if self.action_names and action < len(self.action_names):
            return self.action_names[action]
        return f"A{action}"
    
    def _format_reward(self, reward: float) -> str:
        if reward > 0:
            return f"+{reward:.1f}"
        elif reward < 0:
            return f"{reward:.1f}"
        else:
            return "0"


class ParallelHistoryBuffer:
    """Manage history buffers for multiple parallel environments"""
    
    def __init__(
        self,
        num_envs: int,
        history_length: int = 5,
        action_names: Optional[List[str]] = None,
        template_style: str = "detailed",
    ):
        self.num_envs = num_envs
        self.buffers = [
            HistoryBuffer(
                history_length=history_length,
                action_names=action_names,
                template_style=template_style,
            )
            for _ in range(num_envs)
        ]
    
    def add(self, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, infos: List[Dict]):
        for i in range(self.num_envs):
            self.buffers[i].add(
                action=int(actions[i]),
                reward=float(rewards[i]),
                done=bool(dones[i]),
                info=infos[i] if i < len(infos) else {}
            )
            
            if dones[i]:
                self.buffers[i].reset()
    
    def to_text_batch(self) -> List[str]:
        return [buffer.to_text() for buffer in self.buffers]
    
    def reset_all(self):
        for buffer in self.buffers:
            buffer.reset()
