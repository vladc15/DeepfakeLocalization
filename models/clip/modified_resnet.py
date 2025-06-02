import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any, Union, List


# type of residual block used in ResNet-50, ResNet-101, and ResNet-152 architectures
# https://en.wikipedia.org/wiki/Residual_neural_network#Variants_of_residual_blocks
class Bottleneck(nn.Module):
    expansion: int = 4 # expansion factor for determining the number of output channels at the end of the block

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        # here, all convolutions have stride 1 instead of 2
        # using average pooling of stride 2 - rect-2 blur pooling
        # this should blur the image before downsampling, which is better for CLIP   
        # also known as antialiasing convolutions

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # used for antialiasing (stride > 1 introduces aliasing)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.stride = stride
        
        # sometimes downsampling is needed to match the input and output sizes (in skip connections)
        if stride > 1 or in_channels != out_channels * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                OrderedDict([
                    ("-1", nn.AvgPool2d(stride)),
                    ("0", nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=1, bias=False)),
                    ("1", nn.BatchNorm2d(out_channels * self.expansion)),
                ])
            )
        else:
            self.downsample = None        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x # save the input for the skip connection (identity mapping)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity) # downsample the input to match the output size

        out = self.relu3(out+identity)
        return out


# another modification (and the last one) to the original ResNet architecture in CLIP
# the final convolutional layer is followed by a multiheaded attention layer
# https://en.wikipedia.org/wiki/Contrastive_Language-Image_Pre-training#Image_model

# it is similar to the attention layer in the Transformer architecture
# only here it is used for images, therefore it works with patches of images (also used in ViT)
# this works like a global average pooling layer through attention
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None) -> None:
        super().__init__()

        self.num_heads = num_heads

        # divide by square root of the embedding dimension for stability
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # final projection layer, after the attention mechanism
        # output projection layer
        # in the case output_dim was not specified, we keep it to embedding_dim
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # we need to work with patches of the images
        # (B, C, H, W) -> (B, H*W, C) -> (H*W, B, C)
        x = x.flatten(start_dim=2).permute(2, 0, 1)
                                                    
        # CLS token added in front
        # it should contain the global information about the image
        # (H*W, B, C) -> (H*W + 1, B, C)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)

        # now we add the positional embedding
        x = x + self.positional_embedding[:, None, :].to(x.dtype)

        # now we compute the attention scores
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x, # CLS token is the query, all patches are keys and values
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        # return only the CLS token
        # which contains the global information about the image
        return x.squeeze(0)



# now the modified ResNet architecture used in CLIP
class ModifiedResNet(nn.Module):
    def __init__(self, layers: Union[List[int], Tuple[int, ...]], output_dim: int,
                 heads: int, input_resolution: int = 224, width: int = 64,
                 intermediate_layers: bool = False) -> None:
        super().__init__()

        self.input_resolution = input_resolution
        self.intermediate_layers = intermediate_layers # added for the purpose of extracting the intermediate layers features
        self.output_dim = output_dim

        # 3-layer stem - modified from the original ResNet architecture
        # 3 layers of 3x3 convolutions instead of a single 7x7 convolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # mutable variable that will change to store the number of input channels (in_channels)
        self._inplanes = width

        # residual layers
        # num of channels increases while the spatial resolution decreases
        self.layer1 = self._make_residual_layer(width, layers[0])
        self.layer2 = self._make_residual_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_residual_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_residual_layer(width * 8, layers[3], stride=2)

        # attention pooling layer
        # ResNet's feature dimension will be channels * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, width * 32, heads, output_dim)
        
    def _make_residual_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []

        # first block
        layers.append(Bottleneck(self._inplanes, out_channels, stride))
        self._inplanes = out_channels * Bottleneck.expansion # update the number of input channels

        # remaining blocks
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, out_channels)) # here the in_channels match the out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Union[Dict[str, Any], torch.Tensor]:
        out = {}
        x = x.type(self.conv1.weight.dtype) # convert to the same type as the weights
        
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)

        # residual layers
        x = self.layer1(x)
        out["layer1"] = x
        x = self.layer2(x)
        out["layer2"] = x
        x = self.layer3(x)
        out["layer3"] = x
        x = self.layer4(x)
        out["layer4"] = x

        # attention pooling
        x = self.attnpool(x)
        out["final_embeddings"] = x

        if self.intermediate_layers:
            return out
        else:
            return x

            