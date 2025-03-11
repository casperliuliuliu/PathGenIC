import torch
import torch.nn as nn
import torch.nn.functional as F

def adaptivate_pooling(images, target_tokens):
    batch_size, _, feature_dim = images.shape
    first_token = images[:, 0:1, :]  # Shape: (batch_size, 1, feature_dim)
    other_tokens = images[:, 1:, :]  # Shape: (batch_size, num_token - 1, feature_dim)
    pooled_images = F.adaptive_avg_pool1d(other_tokens.permute(0, 2, 1), target_tokens - 1)  # Shape: (batch_size, 1024, 199)
    pooled_images = pooled_images.permute(0, 2, 1)  # Shape: (batch_size, 199, 1024)
    images = torch.cat([first_token, pooled_images], dim=1)
    return images

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + feedforward_output)
        return x.transpose(0, 1)

class AVG_AttentionModel(nn.Module):
    def __init__(self, num_blocks=4, embed_dim=1024, num_heads=8, dropout=0.1, output_dim=1024):
        super(AVG_AttentionModel, self).__init__()
        self.attention_blocks = nn.ModuleList(
            [AttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_blocks)]
        )
        self.fc = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x):
        x = x.to(self.attention_blocks[0].attention.in_proj_weight.dtype)
        
        for block in self.attention_blocks:
            x = block(x)
        x = adaptivate_pooling(x, 200)
        x = self.fc(x)
        return x

class token_AttentionModel(nn.Module):
    def __init__(self, num_blocks=4, embed_dim=1024, num_heads=8, dropout=0.1, output_dim=1024, num_token=200):
        super(token_AttentionModel, self).__init__()
        self.attention_blocks = nn.ModuleList(
            [AttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_blocks)]
        )
        self.fc = nn.Linear(embed_dim, output_dim)
        self.token = nn.Parameter(torch.randn(num_token, embed_dim))
        self.num_token = num_token

    def forward(self, x):
        x = x.to(self.attention_blocks[0].attention.in_proj_weight.dtype)
        x = torch.cat([x, self.token.repeat(x.size(0), 1, 1)], dim=1)
        for block in self.attention_blocks:
            x = block(x)
        x = x[:, -self.num_token:, :]
        x = self.fc(x)
        return x

def build_casper_attention(config):
    if config.model == "avg":
        return AVG_AttentionModel(num_blocks=config.num_blocks, embed_dim=config.embed_dim, num_heads=config.num_heads, dropout=config.dropout, output_dim=config.output_dim)
    elif config.model == "token":
        print("===============================")
        print("Building token model for casper")
        return token_AttentionModel(num_blocks=config.num_blocks, embed_dim=config.embed_dim, num_heads=config.num_heads, dropout=config.dropout, output_dim=config.output_dim, num_token=config.num_token)


if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(**{
        "model": "token",
        "num_blocks": 2,
        "embed_dim": 128,
        "num_heads": 8,
        "dropout": 0.1,
        "output_dim": 1024,
        "num_token": 400,
    })

    model = build_casper_attention(config)
    print(model)
    input_tensor = torch.randn(4, 800, 128)
    output = model(input_tensor)
    print(output.shape)