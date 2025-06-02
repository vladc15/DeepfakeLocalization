import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any, Union, List

# define layer norm that works with lower precision as well
# pre-trained models come with float16 weights
# training and predicting with float16 is faster and uses less memory as well
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        initial_type = x.dtype
        res = super().forward(x.type(torch.float32))
        return res.type(initial_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # d_model is the dimension of the features given 
        self.attn = nn.MultiheadAttention(d_model, heads)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # used for masking the attention weights
        # and restricting attention between certain tokens
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.resblocks(x) # this is the original code

        # we need to extract the features
        out = {}
        for i, layer in enumerate(self.resblocks.children()):
            x = layer(x)

            # save the features after each layer
            # shape: (seq_len, batch_size, width), where
            # seq_len = number of tokens in the input sequence
            # width = dimension of the features
            # x[0] contains the CLS token - which can be used for global features
            out[f"layer{i}"] = x
        return out, x
    

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, intermediate_layers: bool = False):
        super().__init__()

        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.intermediate_layers = intermediate_layers # flag whether to return the features after each layer or not

        # dividing the input image into patches for visual tokenization
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # scale features to avoid gradient explosion
        # width is d_model, which is the dimension of the features given to the transformer
        # i.e. the token embedding dimension
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.conv1(x) # shape: (batch_size, width, grid=input_resolution // patch_size, grid=input_resolution // patch_size)
        x = x.reshape(x.shape[0], x.shape[1], -1) # shape: (batch_size, width, grid * grid); similar to usual text transformers, obtaining a sequence of tokens
        x = x.permute(0, 2, 1) # shape: (batch_size, grid * grid, width)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) # shape: (batch_size, grid * grid + 1, width)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2) # shape: (grid * grid + 1, batch_size, width); (seq_len, batch_size, width)
        out, x = self.transformer(x)
        x = x.permute(1, 0, 2) # back to (batch_size, grid * grid + 1, width); (batch_size, seq_len, width)
        x = self.ln_post(x[:, 0, :]) # shape: (batch_size, width); CLS token

        # we are only interested in the CLS token
        # they keep the transformer structure, hence the CLS (classification token)
        # https://github.com/google-research/vision_transformer/issues/83#issuecomment-805661088

        out['before_projection'] = x

        if self.proj is not None:
            x = x @ self.proj # shape: (batch_size, output_dim); projection to the output dimension from x's width
        out['after_projection'] = x

        if self.intermediate_layers: # return all intermediate features, as well as the final CLIP output
            return out
        else: # return only the final CLIP output
            return x