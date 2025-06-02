import torch.nn as nn
import torch
import torch.nn.functional as F
import re
from typing import Optional, Union, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from parameters import Parameters
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAMLocalisationModel(nn.Module):
    def __init__(self, checkpoint_path: str, config_path: str, feature_maps: str, decoder_type: str, device: torch.device) -> None:
        super(SAMLocalisationModel, self).__init__()

        self.feature_maps = feature_maps # this will be one of ['image_embed', 'high_res_feats_0', 'high_res_feats_1']
        self.decoder_type = decoder_type # this will follow the DeCLIP types of decoder, e.g. 'conv-4', 'conv-12', 'conv-20'

        # prepare the encoder
        self.sam_model = build_sam2(config_path, checkpoint_path, device=device)
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)
        
        self.verbose = False

        self._set_decoder()

    def _set_decoder(self) -> None:
        upscaling_layers = []
        filter_sizes = self._get_conv_filter_sizes() # get the filter sizes for the decoder
        num_convs = int(re.search(r'\d+', self.decoder_type).group()) # this represents 4M => M sub-blocks (e.g. conv-20 => M=4)

        for i in range(1, len(filter_sizes)):
            upscaling_layers.append(nn.Conv2d(filter_sizes[i-1], filter_sizes[i], kernel_size=5, padding=2))
            upscaling_layers.append(nn.BatchNorm2d(filter_sizes[i]))
            upscaling_layers.append(nn.ReLU())
            for _ in range(num_convs//4 - 1): # M sub-blocks
                upscaling_layers.append(nn.Conv2d(filter_sizes[i], filter_sizes[i], kernel_size=5, padding=2))
                upscaling_layers.append(nn.BatchNorm2d(filter_sizes[i]))
                upscaling_layers.append(nn.ReLU())

            # skip some upscaling layers if the input is too large (in their case for RN50)
            skip_upscaling = (
                self.feature_maps == "high_res_feats_1" and i == 1
                or self.feature_maps == "high_res_feats_0" and i <= 2
                )
            if skip_upscaling:
                continue

            upscaling_layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))

        # in their case, they needed the output to be 256x256, but RN50 outputs 224x224
        upscaling_layers.append(nn.Upsample(size=(256, 256), mode='bilinear'))

        # upscaling_layers.append(nn.Conv2d(16, 1, kernel_size=5, padding=2))
        upscaling_layers.append(nn.Conv2d(64, 1, kernel_size=5, padding=2))

        self.fc = nn.Sequential(*upscaling_layers)


    def _get_conv_filter_sizes(self) -> List[int]:
        # get the descending number of filters (in_channels/out_channels) for the decoder
        # image embeddings sizes:
        # image_embed: 256x64x64 (1x256x64x64)
        # high_res_feats_0: 32x256x256 (1x32x256x256)
        # high_res_feats_1: 64x128x128 (1x64x128x128)
        if self.feature_maps == 'image_embed':
            return [256, 128, 64]
        elif self.feature_maps == 'high_res_feats_0':
            return [32, 64]
        elif self.feature_maps == 'high_res_feats_1':
            return [64, 64]


    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        # first set the image tensor for encoding
        # max_val = x.max().item()
        # is_scaled_tensor = max_val <= 1.0 # check if the tensor is scaled between 0 and 1
        x_np = x.permute(0, 2, 3, 1).cpu().detach().numpy() # convert to numpy array, as SAM2ImagePredictor expects numpy array
        # x_np = x_np.transpose(0, 2, 3, 1) # convert to HWC format (height, width, channels)
        # x_np = x_np.astype('float32')
        # if is_scaled_tensor:
        #     x_np *= 255.0
        # x_np = x_np.astype('uint8') # convert to uint8 format
        x_list = [x_np[i] for i in range(x_np.shape[0])] # convert to list of numpy arrays

        self.sam_predictor.set_image_batch(x_list)

        if self.feature_maps == 'image_embed':
            return self.sam_predictor._features['image_embed'] # batch_sizex256x64x64
        elif self.feature_maps == 'high_res_feats_0':
            return self.sam_predictor._features['high_res_feats'][0] # batch_sizex32x256x256
        elif self.feature_maps == 'high_res_feats_1':
            return self.sam_predictor._features['high_res_feats'][1] # batch_sizex64x128x128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get the feature maps from the SAM model
        feature_maps = self.feature_extractor(x)

        # plot the feature maps
        if self.verbose:
            feature_map = feature_maps[0].cpu().detach()  # shape: (C, H, W)
            num_channels = 1  # first 16
            plt.figure(figsize=(12, 12))
            
            for i in range(num_channels):
                plt.subplot(4, 4, i + 1)
                plt.imshow(feature_map[i], cmap='viridis')
                plt.axis('off')
                plt.title(f'Ch {i}')
            
            plt.tight_layout()
            plt.show()


        # pass through the decoder layers
        output = self.fc(feature_maps)
        
        # show here as well
        if self.verbose:
            masks = output[0].cpu().detach()  # pick first item in batch
            num_masks = min(16, masks.shape[0])  # limit to 16 channels
            plt.figure(figsize=(12, 12))
    
            for i in range(num_masks):
                plt.subplot(4, 4, i + 1)
                plt.imshow(masks[i].cpu().detach().to(dtype=torch.float32).numpy(), cmap='inferno')
                plt.axis('off')
                plt.title(f'Mask {i}')
            
            plt.tight_layout()
            plt.show()

        output = torch.flatten(output, start_dim=1) # flatten the output for the prediction
        return output



def get_sam_model(params: Parameters) -> SAMLocalisationModel:
    assert params.sam_checkpoint_path is not None, "Checkpoint path must be provided."
    assert params.sam_config_path is not None, "Config path must be provided."
    assert params.feature_layer in ['image_embed', 'high_res_feats_0', 'high_res_feats_1'], "Feature layer must be one of ['image_embed', 'high_res_feats_0', 'high_res_feats_1']."
    assert params.decoder_type is not None, "Decoder type must be provided."

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return SAMLocalisationModel(
        checkpoint_path=params.sam_checkpoint_path,
        config_path=params.sam_config_path,
        feature_maps=params.feature_layer,
        decoder_type=params.decoder_type,
        device=device
    )