import torch.nn as nn
import torch
import torch.nn.functional as F
import re
from typing import Optional, Union, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from models.clip import load_clip
from parameters import Parameters


# using CLIP for detection / classification of the deepfake images
class CLIPModelDetection(nn.Module):
    def __init__(self, name: str, intermediate_layer_output: str = None, num_classes: int = 2) -> None:
        super(CLIPModelDetection, self).__init__()
        
        self.name = name # CLIP architecture
        self.intermidiate_layer_output = intermediate_layer_output
        self.num_classes = num_classes

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
        
        feature_dim = self.model.visual.output_dim # we can access output_dim directly to get the dim

        upscaling_layers.append(nn.Linear(feature_dim, self.num_classes))

        self.fc = nn.Sequential(*upscaling_layers)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # extract features - encoder part
        features = self.model.visual(x)['after_projection'].flatten(1) # we are not only interested in using the CLS token (even though it contains the most information), 
                                                   # but the rest of the tokens as well (since they contain more information about the details of the image, resulting in a better representation)
        # print(self.model.visual(x)) # after the modifs of the output (having the feature maps extraction now), model.visual(x) is a dict of the feature maps, from whom we chose the ones at the end

        output = self.fc(features)
        
        return output
    

def get_detection_model(params: Parameters) -> nn.Module:
    assert params.arch in ["CLIP:RN50", "CLIP:ViT-L/14", "CLIP:ViT-L/14,RN50"] # originally proposed architectures
    detection_model = CLIPModelDetection(params.arch.split(':')[1], 
                                 intermediate_layer_output=params.feature_layer, 
                                 # decoder_type=params.decoder_type, # only a linear layer is used
                                 num_classes=2 if params.task_type == 'detection' else 5, # either 2 for detection or 5 for classification
                                 )

    if params.checkpoint_path == '':
        return detection_model
    else: # for continuing training
        state_dict = torch.load(params.checkpoint_path, map_location='cpu')
        detection_model.load_state_dict(state_dict, strict=False)
        detection_model = detection_model.to("cuda")
        if params.data_label == 'train':
            detection_model.train()
        else:
            detection_model.eval()
        return detection_model
