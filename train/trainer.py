import os
import torch
import torch.nn as nn
import time

from models.declip import get_model
from models.sam_localization import get_sam_model
from models.declip_detection import get_detection_model
from utils.utils import compute_batch_iou, compute_batch_localization_f1, compute_batch_ap, compute_accuracy_detection, compute_average_precision_detection


class Trainer(nn.Module):
    def __init__(self, params):
        super(Trainer, self).__init__()
        self.params = params
        if 'localization' in params.task_type: # fully_supervised_localization or weakly_supervised_localization
            if 'SAM' in params.arch:
                self.model = get_sam_model(params)
            else:
                self.model = get_model(params)
        elif params.task_type == 'detection' or params.task_type == 'classification':
            self.model = get_detection_model(params)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_steps = 0 # internal counter for the number of steps (iterations) done so far

        self.model.to(self.device)

        # initialize parameters in the decoder (all layers)
        for fc in self.model.fc:
            try:
                torch.nn.init.normal_(fc.weight.data, 0.0, params.init_gain)
            except:
                pass

        # we want to only train the decoder (encoder is frozen)
        trainable_params = []
        for name, parameter in self.model.named_parameters():
            if "fc" in name and "resblock" not in name: # train decoder layers
                trainable_params.append(parameter)
            else:
                parameter.requires_grad = False # freeze the parameters of the backbone

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=params.lr,
            betas=(params.beta1, 0.999),
            weight_decay=params.weight_decay
        )
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_type = params.loss_type if hasattr(params, 'loss_type') else 'bce'
        self.pos_weights = None
        
        self.weakly_supervised_label_comparison_type = params.weakly_supervised_label_comparison_type if hasattr(params, 'weakly_supervised_label_comparison_type') else None

        # schedulers - not used in the paper code; they manually reduce the lr
        self.scheduler_steplr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1) # learning rate scheduler
        self.scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True) # learning rate scheduler
        
        if params.task_type == 'detection' or params.task_type == 'classification':
            self.logits = []
            self.labels = []
        else:
            self.ious = []
            self.f1_best = []
            self.f1_fixed = []
            self.ap = []

            self.localization_logits = []
            self.localization_labels = []

    def compute_pos_weight_per_dataset(self, dataloader):
        if self.pos_weights is not None:
            return self.pos_weights
            
        total_pos = 0
        total_neg = 0
        for batch in dataloader:
            inputs, targets, _, _ = batch
            targets = targets.float()
            total_pos += targets.sum().item()
            total_neg += targets.numel() - targets.sum().item()
        self.pos_weights = torch.tensor(total_neg / (total_pos + 1e-6))
        return self.pos_weights

    def compute_pos_weight_per_batch(self, targets):
        num_pos = targets.sum()
        num_neg = targets.numel() - num_pos
        pos_weight = num_neg / (num_pos + 1e-6)
        return pos_weight

    def bce_dice_loss(self, preds, targets):
        #pos_weight = self.compute_pos_weight(targets).to(self.device)
        if self.pos_weights is None:
            pos_weight = self.compute_pos_weight(targets).to(self.device)
        else:
            pos_weight = self.pos_weights
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(preds, targets)
        
        preds_sigmoid = torch.sigmoid(preds)
        preds_flat = preds_sigmoid.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (preds_flat * targets_flat).sum(1)
        dice_loss = 1 - ((2. * intersection + 1e-6) / (preds_flat.sum(1) + targets_flat.sum(1) + 1e-6))
        dice_loss = dice_loss.mean()
        
        return bce_loss + dice_loss

    def focal_loss(self, preds, targets, gamma=2.0):
        #pos_weight = self.compute_pos_weight(targets).to(self.device)
        if self.pos_weights is None:
            pos_weight = self.compute_pos_weight(targets).to(self.device)
        else:
            pos_weight = self.pos_weights
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')(preds, targets)
        
        probs = torch.sigmoid(preds)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t) ** gamma
        
        loss = focal_factor * bce_loss
        return loss.mean()

    def loss_fn(self, preds, targets):
        if self.loss_type == "bce_dice":
            return self.bce_dice_loss(preds, targets)
        elif self.loss_type == "focal":
            return self.focal_loss(preds, targets)
        elif self.loss_type == "bce":
            return nn.BCEWithLogitsLoss()(preds, targets)
        elif self.loss_type == "cross-entropy":
            return nn.CrossEntropyLoss()(preds, targets.long())
        elif self.loss_type == "bce_simple":
            return nn.BCELoss()(preds, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    
    def save_model(self, save_state_filename):
        save_path = os.path.join(self.params.save_dir_models, save_state_filename)
        
        # serialize the current state to a dict
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'feature_layer': self.params.feature_layer,
            'decoder_type': self.params.decoder_type,
        }
        torch.save(state_dict, save_path)

    def set_input(self, input_data):
        self.input = input_data[0].to(self.device)
        self.label = input_data[1].to(self.device).float()

    def forward(self):
        self.output = self.model(self.input)

        # process output
        if 'localization' in self.params.task_type:
            # resize prediction to ground truth mask size
            if self.params.task_type == 'fully_supervised_localization':
                if self.label.size()[1] != 256 * 256:
                    label_size = (int(self.label.size()[1] ** 0.5), int(self.label.size()[1] ** 0.5))
                    self.output = self.output.view(-1, 1, 256, 256)
                    self.output = torch.nn.functional.interpolate(self.output, size=label_size, mode='bilinear', align_corners=False)
                    self.output = torch.flatten(self.output, start_dim=1).unsqueeze(1)
            else:
                self.output = self.output.view(-1, 1, 256, 256)
                self.output = torch.nn.functional.interpolate(self.output, size=(256, 256), mode='bilinear', align_corners=False)
                self.output = torch.flatten(self.output, start_dim=1).unsqueeze(1)
        elif self.params.task_type == 'detection' or self.params.task_type == 'classification': 
            # self.output = torch.mean(self.output, dim=1) # mean over the channels (RGB) to get a single channel output
            pass

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.0
            if param_group['lr'] < min_lr:
                return False # return True or False on whether lr is below or above min_lr
        return True # we can stop training if the lr is below min_lr

    def optimize_parameters(self):
        self.model.train() # set the model to training mode
        # start_time = time.time()
        self.forward()
        # end_time_forward = time.time()

        if 'localization' in self.params.task_type: # fully-supervised or weakly-supervised localization
            if self.params.task_type == 'fully_supervised_localization':
                sigmoid_outputs = torch.sigmoid(self.output)
            
                # unflatten outputs and ground truth masks
                sigmoid_outputs = sigmoid_outputs.view(sigmoid_outputs.size(0), int(sigmoid_outputs.size(1)**0.5), int(sigmoid_outputs.size(1)**0.5))
                labels = self.label.view(self.label.size(0), int(self.label.size(1)**0.5), int(self.label.size(1)**0.5))
    
                batch_iou = compute_batch_iou(sigmoid_outputs, labels)
                self.ious.extend(batch_iou)
    
                f1_best, f1_fixed = compute_batch_localization_f1(sigmoid_outputs, labels)
                self.f1_best.extend(f1_best)
                self.f1_fixed.extend(f1_fixed)
    
                average_precision = compute_batch_ap(sigmoid_outputs, labels)
                self.ap.extend(average_precision)
    
                # compute metrics later, after going through the whole dataset
                
                # self.localization_logits.append(self.output)
                # self.localization_labels.append(self.label)
                
            elif self.params.task_type == 'weakly_supervised_localization':
                sigmoid_outputs = torch.sigmoid(self.output)
            
                # unflatten outputs and ground truth masks
                sigmoid_outputs = sigmoid_outputs.view(sigmoid_outputs.size(0), int(sigmoid_outputs.size(-1)**0.5), int(sigmoid_outputs.size(-1)**0.5))
                
                self.output, self.label = self.label_comparison(sigmoid_outputs, self.label, self.weakly_supervised_label_comparison_type)


                # self.localization_logits.append(self.output)
                # self.localization_labels.append(self.label)

        else: # detection or classification
            self.logits.append(self.output)
            self.labels.append(self.label)

        self.optimizer.zero_grad()
        self.loss = self.loss_fn(self.output, self.label)
        self.loss.backward()
        self.optimizer.step()
        
    def label_comparison(self, pred, label, comparison_type):
        # pred should contain probabilities, after applying sigmoid
        B, H, W = pred.shape
        
        if comparison_type == 'expansion': # expand label on the whole image
            label = label.view(B, 1, 1).expand(B, H, W)
            return pred, label # here BCEWithLogitsLoss can be applied, just like in the fully-supervised scenario
            
        elif comparison_type == 'max_pooling':
            pred = pred.view(B, -1) # flatten back
            pred_max = pred.max(dim=1).values # extract max pixel probability values
            return pred_max, label # here we can use BCELoss (unlike fully-supervised, we applied sigmoid; unlike detection, we use single channel probability, so no CrossEntropy)
            
        elif comparison_type == 'avg_pooling':
            pred_mean = pred.mean(dim=(1, 2)) # (B,)
            pred_mean = pred_mean.clamp(min=1e-6) # against log(0)
            pred = torch.log(pred_mean) + 0.7 # mean should be around 0.5 and we need to get the value to positives
            return pred, label # here we can use BCELoss (unlike fully-supervised, we applied sigmoid; unlike detection, we use single channel probability, so no CrossEntropy)
            

    def format_output_detection(self):
        if self.params.task_type == 'detection' or self.params.task_type == 'classification':
            self.logits = torch.cat(self.logits, dim=0)
            self.labels = torch.cat(self.labels, dim=0)
        elif 'localization' in self.params.task_type:
            self.localization_logits = torch.cat(self.localization_logits, dim=0)
            self.localization_labels = torch.cat(self.localization_labels, dim=0)