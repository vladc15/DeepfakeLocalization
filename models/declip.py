import torch.nn as nn
import torch
import torch.nn.functional as F
import re
from typing import Optional, Union, List, Tuple

from models.clip import load_clip
from parameters import Parameters


# the DeCLIP model according to the paper and implementation at https://github.com/bit-ml/DeCLIP
class CLIPModelLocalisation(nn.Module):
    def __init__(self, name: str, intermediate_layer_output: str = None, decoder_type: str = "conv-4") -> None:
        super(CLIPModelLocalisation, self).__init__()
        
        self.name = name # CLIP architecture
        self.decoder_type = decoder_type
        self.intermidiate_layer_output = intermediate_layer_output

        self._set_backbone()
        self._set_decoder()

    def _set_backbone(self) -> None:
        if self.name in ["RN50", "ViT-L/14"]: # CLIP architectures
            self.model, _ = load_clip.load(self.name, device="cpu", intermediate_layers = (self.intermidiate_layer_output != None))
            self.model = self.model.to("cuda") # we can let the model load on the CPU and then move it to GPU for float patching
        elif "RN50" in self.name and "ViT-L/14" in self.name: # ViT+RN - combine both architectures
            name = self.name.split(",")
            model1, _ = load_clip.load(name[0], device="cpu", intermediate_layers = (self.intermidiate_layer_output != None)) 
            model2, _ = load_clip.load(name[1], device="cpu", intermediate_layers = (self.intermidiate_layer_output != None))            
            self.model = [model1.to("cuda"), model2.to("cuda")]

    def _set_decoder(self) -> None:
        upscaling_layers = []

        # feature maps will be of size 16 x 16 x 1024 for ViT-L/14 after reshaping
        # most experiments are done using ViT-L/14; for ViT+RN fusion, we need 2048 channels / initial filter_size
        # which is specified for the convolutional decoder (not the rest of them)

        # localization with convolutional layers
        filter_sizes = self._get_conv_filter_sizes(self.name, self.intermidiate_layer_output, self.decoder_type)
        num_convs = int(re.search(r'\d{0,3}$', self.decoder_type).group()) # conv-20 => 20 convs (authors vary between 4, 12 and 20 convs)
                                                                            # this represents 4M => M sub-blocks

        for i in range(1, len(filter_sizes)): # 4 blocks in total
            upscaling_layers.append(nn.Conv2d(filter_sizes[i-1], filter_sizes[i], kernel_size=5, padding=2))
            upscaling_layers.append(nn.BatchNorm2d(filter_sizes[i]))
            upscaling_layers.append(nn.ReLU())
            for _ in range(num_convs//4 - 1): # M sub-blocks
                upscaling_layers.append(nn.Conv2d(filter_sizes[i], filter_sizes[i], kernel_size=5, padding=2))
                upscaling_layers.append(nn.BatchNorm2d(filter_sizes[i]))
                upscaling_layers.append(nn.ReLU())

            # skip some upscaling layers if the input is too large (case for CNNs - RN50)
            # manually set the upscaling layers for large channels
            skip_upscaling = (
                self.intermidiate_layer_output == "layer2" and i == 1
                or self.intermidiate_layer_output == "layer1" and i <= 2
                ) and ("RN50" in self.name) # they also tried Xception, but it was performing worse than RN50
            if skip_upscaling:
                continue

            upscaling_layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))

        # CNNs output may not be in (256, 256) - usually a (224, 224) size
        if "RN50" in self.name:
            upscaling_layers.append(nn.Upsample(size=(256, 256), mode='bilinear'))

        upscaling_layers.append(nn.Conv2d(64, 1, kernel_size=5, padding=2))

        self.fc = nn.Sequential(*upscaling_layers)

    def _get_conv_filter_sizes(self, name: str, intermidiate_layer_output: str, decoder_type: str) -> List[int]:
        if "RN50" in name and "ViT-L/14" in name:
            return [1024*2, 512, 256, 128, 64] # combines ViT (L21) with RN50 (L3), where both have 1024 channels
        elif "RN50" in name: # num of channels based on our extracted features in ResNet
            if intermidiate_layer_output == "layer1":
                return [256, 512, 256, 128, 64]
            elif intermidiate_layer_output == "layer2":
                return [512, 512, 256, 128, 64]
            elif intermidiate_layer_output == "layer3":
                return [1024, 512, 256, 128, 64]
            elif intermidiate_layer_output == "layer4":
                return [2048, 512, 256, 128, 64]
        else: # ViT-L/14 gives 1024 channels for its embeddings (layer 21)
            return [1024, 512, 256, 128, 64]
    
    def _feature_map_transform(self, features):
        features_transformed = features.permute(1, 2, 0)
        features_transformed = features_transformed.view(features_transformed.size()[0], features_transformed.size()[1], int(features_transformed.size()[2]**0.5), int(features_transformed.size()[2]**0.5))
        return features_transformed # shape: (batch_size, channels, height, width), e.g. (batch_size, 1024, 16, 16) for ViT-L/14

    def feature_extraction(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "RN50" or self.name == "ViT-L/14":
            features = self.model.encode_image(x) # will contain the 'out' with all the layers features
            if self.intermidiate_layer_output: # specify the layer to extract features from
                features = features[self.intermidiate_layer_output]
            else: # choose the last layer
                if self.name == "RN50":
                    features = features["layer4"]
                else:
                    features = features["layer23"]
        elif "RN50" in self.name and "ViT-L/14" in self.name: # ViT+RN fusion
            # given ViT feature layer - all will be of size 16 x 16 x 1024 for ViT-L/14 after reshaping
            features_vit = self.model[0].encode_image(x)[self.intermidiate_layer_output]
            features_vit = self._feature_map_transform(features_vit[1:])
            # explicit RN50 3rd layer to match the feature dimension
            features_rn50 = self.model[1].encode_image(x)["layer3"]
            features_rn50 = F.interpolate(features_rn50, size=(16, 16), mode='bilinear', align_corners=False) # shape: (batch_size, 1024, 16, 16) to match ViT-L/14
            features = torch.cat([features_vit, features_rn50], 1) # concatenate the features from both architectures
        
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # extract features - encoder part
        features = self.feature_extraction(x)
        
        # decoder part
        if "RN50" in self.name and "ViT-L/14" in self.name:
            output = self.fc(features)

        elif self.name == "RN50":
            output = self.fc(features)

        else: # ViT with conv decoder
            features = features[1:] # exclude CLS token
            output = self._feature_map_transform(features)
            output = self.fc(output)

        output = torch.flatten(output, start_dim =1) # flatten the output to (batch_size, 1, 256, 256) for the final prediction
        return output
    

def get_model(params: Parameters) -> nn.Module:
    assert params.arch in ["CLIP:RN50", "CLIP:ViT-L/14", "CLIP:ViT-L/14,RN50"] # originally proposed architectures

    localisation_model = CLIPModelLocalisation(params.arch.split(':')[1], 
                                 intermediate_layer_output=params.feature_layer, 
                                 decoder_type=params.decoder_type)

    if params.checkpoint_path == '':
        return localisation_model
    else: # for continuing training
        state_dict = torch.load(params.checkpoint_path, map_location='cpu')
        localisation_model.load_state_dict(state_dict, strict=False)
        localisation_model = localisation_model.to("cuda")
        if params.data_label == 'train':
            localisation_model.train()
        else:
            localisation_model.eval()
        return localisation_model
