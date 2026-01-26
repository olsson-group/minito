import torch.nn as nn
import numpy as np
import torch

"""
This code is in early development and should not be expected to work out-of-the-box.
"""

class PositionalEncoder:
    def __init__(self, dim, max_length):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for positional encoding for sin/cos"

        self.dim = dim
        self.max_length = max_length
        self.max_rank = dim // 2

    def forward(self, x):
        encodings = [self.positional_encoding(x, rank) for rank in range(1, self.max_rank + 1)]
        encodings = torch.cat(
            encodings,
            axis=1,
        )
        return encodings

    def positional_encoding(self, x, rank):
        sin = torch.sin(x / self.max_length * rank * np.pi)
        cos = torch.cos(x / self.max_length * rank * np.pi)
        #assert cos.device == self.device, f"batch device {cos.device} != model device {self.device}"
        return torch.stack((cos, sin), axis=1)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 64),
            torch.nn.LayerNorm(64),
            torch.nn.SiLU(),
            nn.Linear(64, 64),         
            torch.nn.LayerNorm(64),
            torch.nn.SiLU(),
            nn.Linear(64, 64),         
            torch.nn.LayerNorm(64),
            torch.nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.posencoder = PositionalEncoder(dim=4, max_length=4)
    
    def forward(self, t, batch, xt):
        input = torch.cat([batch['cond']['x'], xt, t.unsqueeze(1), self.posencoder.forward(batch['cond']['beta']), self.posencoder.forward(batch['lag'])], dim=1)
        return self.net(input)
    


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.net(x))

class ImprovedVectorField(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=256, num_blocks=3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.posencoder = PositionalEncoder(dim=4, max_length=4)

        
        self.final_proj = nn.Linear(hidden_dim, 1)

    def forward(self, t, batch, xt):
       
        raw_input = torch.cat([
            batch['cond']['x'], 
            xt, 
            t.unsqueeze(1), 
            self.posencoder.forward(batch['cond']['beta']), 
            self.posencoder.forward(batch['lag'])
        ], dim=1)

        h = self.input_proj(raw_input)
        
        t_emb = self.t_embed(t.unsqueeze(1))
        
        for block in self.blocks:
            h = block(h + t_emb) 
            
        return self.final_proj(h)
    

class ResBlockFiLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x, scale, shift):
        h = self.norm(x)
        h = h * (1 + scale) + shift 
        h = self.linear1(self.act(h))
        h = self.linear2(self.act(h))
        return x + h

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=10.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(1, embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x @ self.W * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class UltraVectorField(nn.Module):
    def __init__(self, hidden_dim=256, num_blocks=4):
        super().__init__()
        self.coord_encoder = GaussianFourierProjection(embed_dim=64, scale=16.0)
        
        self.cond_encoder = PositionalEncoder(dim=16, max_length=10)
        
        self.input_proj = nn.Linear(161, hidden_dim)
        
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2) # Output scale & shift
        )
        
        self.blocks = nn.ModuleList([ResBlockFiLM(hidden_dim) for _ in range(num_blocks)])
        self.final_proj = nn.Linear(hidden_dim, 1)

    def forward(self, t, batch, xt):
        x_enc = self.coord_encoder(batch['cond']['x'])
        xt_enc = self.coord_encoder(xt)
        beta_enc = self.cond_encoder.forward(batch['cond']['beta'])
        lag_enc = self.cond_encoder.forward(batch['lag'])
        
        raw_input = torch.cat([x_enc, xt_enc, t.unsqueeze(1), beta_enc, lag_enc], dim=1)
        
        h = self.input_proj(raw_input)
        
        t_params = self.t_embed(t.unsqueeze(1))
        scale, shift = torch.chunk(t_params, 2, dim=1)
        
        for block in self.blocks:
            h = block(h, scale, shift)
            
        return self.final_proj(h)