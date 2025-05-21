import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorFieldNet(nn.Module):
    def __init__(
        self,
        dim: int = 6,
        condition_dim: int = 6,
        embed_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        time_horizon: int = 60,
        action_horizon: int = 50
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.time_mlp = nn.Linear(time_horizon, embed_dim)
        self.pos_embed = nn.Linear(dim * time_horizon, embed_dim)
        self.condition_mlp = nn.Linear(condition_dim * time_horizon, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(embed_dim, 2 * self.action_horizon)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, condition_mask: torch.Tensor):
        """
        params:
            x: [B, x_dim, H]
            t: [B, time_dim, H]
            condition: [B, cond_dim, H, N]
            condition_mask: [B, H, N] (1 indicates valid, 0 indicates invalid)
        returns:
            [B, x_dim * action_horizon]
        """
        B, _, H, T = condition.shape
        masked_condition = condition * condition_mask[..., None, :]  # [B, cond_dim, H, N]
        masked_condition = masked_condition.view(B, -1, T) # [B, cond_dim * H, N]
        masked_condition = self.condition_mlp(masked_condition)
        
        x = x.view(B, -1)       # [B, x_dim * H]
        t = t.view(B, -1)       # [B, time_dim * H]
        x = self.pos_embed(x)
        t = self.time_mlp(t)
        x = torch.cat([x[..., None], t[..., None], masked_condition], dim=1)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        
        x = self.out_proj(x)
        return x



if __name__ == "__main__":
    model = VectorFieldNet()
    x = torch.randn(10, 6, 60)
    t = torch.randn(10, 1, 60)
    condition = torch.randn(10, 6, 60, 10)
    condition_mask = torch.randn(10, 60, 10)
    print(model(x, t, condition, condition_mask).shape)
    
