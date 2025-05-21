import torch
import torch.nn as nn
from typing import List, Dict


class FlowMatcher:
    def __init__(self, model: nn.Module, action_dim: int, action_horizon: int):
        self.model = model
        self.action_dim = action_dim
        self.action_horizon = action_horizon
    
    @torch.no_grad()
    def preprocess_observation(self, observations: Dict) -> torch.Tensor:
        history = observations["hist"]
        history_mask = observations["hist_mask"]
        ...
        
    @torch.no_grad()
    def compute_loss(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        params:
            observations: [B, x_dim, H]
            actions: [B, x_dim, H]
        returns:
            [B]
        """
    
    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        params:
            observations: [B, x_dim, H]
        returns:
            [B, x_dim, H]
        """
        batch_size = observations.shape[0]
        noise = torch.randn((batch_size, self.action_horizon, self.action_dim), device=observations.device)
        dt = 1.0 / num_steps
        x_t = noise
        t = torch.zeros((batch_size, 1), device=observations.device)
        
        for _ in range(num_steps):
            x_t = x_t + dt * self.model(x_t, t, observations)
            t = t + dt
        
        return x_t