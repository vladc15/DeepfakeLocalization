import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms

from models.declip import get_model

class VotingEnsembleModel:
    def __init__(self, models_params, state_dicts, voting_method='soft', threshold=0.5, weights=None):
        assert len(models_params) == len(state_dicts), "Number of models and state dicts must match."
        
        self.models = [None for _ in range(len(models_params))]
        for i, model_params in enumerate(models_params):
            trained_model = get_model(model_params)
            trained_model.load_state_dict(state_dicts[i]['model'], strict=False)
            trained_model.eval()
            trained_model.to('cuda')
            self.models[i] = trained_model
            
        self.voting_method = voting_method # 'soft' or 'hard'
        self.threshold = threshold 
        self.weights = weights if weights is not None else [1] * len(models_params) # for weighted aggregation (can be used for both soft and hard voting)

    def predict(self, images, masks):
        prediction_masks = []

        images = images.to('cuda')
        
        for model in self.models:
            # make the predictions for each model
            predictions = torch.sigmoid(model(images)).squeeze(1)

            # back to mask dimensions
            predictions = predictions.view(predictions.size(0), int(predictions.size(1)**0.5), int(predictions.size(1)**0.5))
            resized_predictions = []
            for i, pred in enumerate(predictions):
                if pred.size() != masks[i].size():
                    pred_resized = F.resize(pred.unsqueeze(0), masks[i].size(), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).squeeze(0)
                    resized_predictions.append(pred_resized)
                else:
                    resized_predictions.append(pred)

            resized_predictions = torch.stack(resized_predictions).to('cuda')
            prediction_masks.append(resized_predictions)

        batch_size = images.size(0)
        height, width = prediction_masks[0].shape[-2:]
        final_prediction = torch.zeros((batch_size, height, width), device='cuda')
        # final_prediction = torch.zeros_like(predictions[0])
        if self.voting_method == 'soft': # average the probabilities from each model
            for i, preds in enumerate(prediction_masks):
                final_prediction += preds * self.weights[i]
            final_prediction /= sum(self.weights) # weighted average of the probabilities
        elif self.voting_method == 'hard': # majority vote
            for i, preds in enumerate(prediction_masks):
                final_prediction += (preds > self.threshold).float() * self.weights[i]
            final_prediction = (final_prediction > (len(self.models) / 2)).float() # more than half votes
        
        return final_prediction.cpu()


class VotingEnsembleModelLoadedBaseModels:
    def __init__(self, models, voting_method='soft', threshold=0.5, weights=None):
        self.models = models
        for i in range(len(models)):
            self.models[i].eval()
            self.models[i].to('cuda')
        self.voting_method = voting_method
        self.threshold = threshold
        self.weights = weights if weights is not None else [1] * len(models)

    def predict(self, images, masks):
        prediction_masks = []

        images = images.to('cuda')
        
        for model in self.models:
            # make the predictions for each model
            predictions = torch.sigmoid(model(images)).squeeze(1)

            # back to mask dimensions
            predictions = predictions.view(predictions.size(0), int(predictions.size(1)**0.5), int(predictions.size(1)**0.5))
            resized_predictions = []
            for i, pred in enumerate(predictions):
                if pred.size() != masks[i].size():
                    pred_resized = F.resize(pred.unsqueeze(0), masks[i].size(), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).squeeze(0)
                    resized_predictions.append(pred_resized)
                else:
                    resized_predictions.append(pred)

            resized_predictions = torch.stack(resized_predictions).to('cuda')
            prediction_masks.append(resized_predictions)

        batch_size = images.size(0)
        height, width = prediction_masks[0].shape[-2:]
        final_prediction = torch.zeros((batch_size, height, width), device='cuda')
        # final_prediction = torch.zeros_like(predictions[0])
        if self.voting_method == 'soft': # average the probabilities from each model
            for i, preds in enumerate(prediction_masks):
                final_prediction += preds * self.weights[i]
            final_prediction /= sum(self.weights) # weighted average of the probabilities
        elif self.voting_method == 'hard': # majority vote
            for i, preds in enumerate(prediction_masks):
                final_prediction += (preds > self.threshold).float() * self.weights[i]
            final_prediction = (final_prediction > (len(self.models) / 2)).float() # more than half votes
        
        return final_prediction.cpu()

