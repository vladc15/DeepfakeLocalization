import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from deepfake_datasets.datasets import MEAN, STD

# this architecture is a 2-step model
# where the first model predicts is either a detector or a classifier
# then the second model(s) predict(s) the localization of the manipulated region
# we can either use a single localization model (generalizer) or multiple localization models (specializers)


class TwoStepModel:
    def __init__(self, detection_model, localization_models, first_step_type='classifier'):
        self.label_map = {
            'real': 0,
            'ldm': 1,
            'repaint': 2,
            'lama': 3,
            'pluralistic': 4,
        }
        self.first_step_type = first_step_type

        self.detection_model = detection_model
        self.detection_model.eval()
        self.detection_model.to('cuda')
        
        self.localization_models = localization_models # should be either a single model or a dict of models
        self.sam_models = False # flag if we used SAM2 models for GANs
        if self.first_step_type == 'detector':
            self.localization_models.eval()
            self.localization_models.to('cuda')
        elif self.first_step_type == 'classifier':
            for deepfake_dataset in self.label_map.keys():
                if deepfake_dataset != 'real':
                    self.localization_models[deepfake_dataset].eval()
                    self.localization_models[deepfake_dataset].to('cuda')
                    
                    if (deepfake_dataset == 'lama' or deepfake_dataset == 'pluralistic') and not self.sam_models:
                        for name, _ in self.localization_models[deepfake_dataset].named_parameters():
                            if 'sam' in name or 'SAM' in name:
                                self.sam_models = True # we used SAM models for localization 
                                break

    
    def predict(self, images):
        # first step: detection/classification
        images = images.to('cuda')
        detection_logits = self.detection_model(images)
        detection_predictions = torch.argmax(detection_logits, dim=1)

        # localization of the manipulated region in the fake images
        # put zero masks for the real images
        final_predictions = [torch.zeros((256, 256)) for _ in range(images.size(0))]

        if self.first_step_type == 'detector':
            indices_fake = torch.where(detection_predictions != self.label_map['real'])[0]
            images_fake = images[indices_fake]
            images_fake = images_fake.to('cuda')

            predictions = torch.sigmoid(self.localization_models(images_fake)).squeeze(1)
            # back to mask dimensions
            predictions = predictions.view(predictions.size(0), int(predictions.size(1)**0.5), int(predictions.size(1)**0.5))
            localization_predictions = []
            for i, pred in enumerate(predictions):
                if pred.size() != torch.Size([256, 256]):
                    pred_resized = F.resize(pred.unsqueeze(0), torch.Size([256, 256]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).squeeze(0)
                    localization_predictions.append(pred_resized)
                else:
                    localization_predictions.append(pred)

            for i, pred in enumerate(localization_predictions):
                final_predictions[indices_fake[i]] = pred

        elif self.first_step_type == 'classifier':
            for deepfake_dataset in self.label_map.keys():
                if deepfake_dataset != 'real':
                    indices_fake_dataset = torch.where(detection_predictions == self.label_map[deepfake_dataset])[0]
                    if indices_fake_dataset.numel() == 0:
                        continue
                    
                    if self.sam_models and (deepfake_dataset == 'lama' or deepfake_dataset == 'pluralistic'): # if we used SAM models, we need to pass images non-normalized
                        inverted_mean = [-m/s for m, s in zip(MEAN, STD)]
                        inverted_std = [1/s for s in STD]
                        sam_reverse_normalization_transform = transforms.Normalize(mean=inverted_mean, std=inverted_std) 
                        images_non_normalized = sam_reverse_normalization_transform(images)
                        images_fake_dataset = images_non_normalized[indices_fake_dataset]
                    else:
                        images_fake_dataset = images[indices_fake_dataset]
                    images_fake_dataset = images_fake_dataset.to('cuda')

                    predictions = torch.sigmoid(self.localization_models[deepfake_dataset](images_fake_dataset)).squeeze(1)
                    # back to mask dimensions
                    predictions = predictions.view(predictions.size(0), int(predictions.size(1)**0.5), int(predictions.size(1)**0.5))
                    localization_predictions = []
                    for i, pred in enumerate(predictions):
                        if pred.size() != torch.Size([256, 256]):
                            pred_resized = F.resize(pred.unsqueeze(0), torch.Size([256, 256]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).squeeze(0)
                            localization_predictions.append(pred_resized)
                        else:
                            localization_predictions.append(pred)

                    for i, pred in enumerate(localization_predictions):
                        final_predictions[indices_fake_dataset[i]] = pred
        
        return [pred.cpu() for pred in final_predictions]