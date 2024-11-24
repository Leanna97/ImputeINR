import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from models import register


class Attention(nn.Module):

    def __init__(self, dim, n_head, head_dim, dropout=0.):
        super().__init__()
        self.n_head = n_head
        inner_dim = n_head * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.scale = head_dim ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, fr, to=None):
        if to is None:
            to = fr
        q = self.to_q(fr)
        k, v = self.to_kv(to).chunk(2, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1) # b h n n
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):

    def __init__(self, dim, ff_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, to=None):
        x = self.norm(x)
        return self.fn(x, to)

class PreNorm1(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, to=None):
        x = self.norm(x)
        return self.fn(x)

# Multi-scale block for extracting features at different temporal resolutions
class MultiScaleBlock(nn.Module):
    def __init__(self, dim, scales=[3, 5, 7]):
        super(MultiScaleBlock, self).__init__()
        self.scales = scales
        # Create a list of convolutional layers to handle different scales
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=s, stride=s, padding=s//2),
                nn.BatchNorm1d(dim),  # Add BatchNorm after each Conv1d layer
                nn.ReLU()  # Optional: ReLU activation after normalization
            )
            for s in scales
        ])
        self.proj = nn.Linear(len(scales) * dim, dim)

    def forward(self, x):
        # Input x: shape (batch_size, seq_len, dim)
        batch_size, seq_len, dim = x.shape
        x = x.permute(0, 2, 1)  # Change to (batch_size, dim, seq_len) for Conv1d

        # Apply convolutions at different scales
        scale_features = [conv(x) for conv in self.convs]

        # Resize the scale features to the same length (can use interpolation or padding)
        scale_features = [F.interpolate(feat, size=seq_len, mode='linear') for feat in scale_features]

        # Concatenate features along the channel dimension (batch_size, dim * len(scales), seq_len)
        multi_scale_features = torch.cat(scale_features, dim=1)
        multi_scale_features = multi_scale_features.permute(0, 2, 1)  # Back to (batch_size, seq_len, dim * len(scales))

        # Project the concatenated features back to the original dim size
        output = self.proj(multi_scale_features)
        return output

@register('transformer_encoder')
class TransformerEncoder(nn.Module):

    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0., multiscale=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.multi_scale_block = MultiScaleBlock(dim)
        self.multiscale = multiscale
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                PreNorm1(dim, FeedForward(dim, ff_dim, dropout=dropout)),
            ]))

    def forward(self, x, to=None):
        if self.multiscale == True:
            x = self.multi_scale_block(x)
        for norm_attn, norm_ff in self.layers:
            x = x + norm_attn(x, to)
            x = x + norm_ff(x)
        return x