import torch
from typing import Any, Dict, List, Tuple, Optional, Union
from PIL import ImageOps, Image
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from models.declip import CLIPModelLocalisation
from models.declip_detection import CLIPModelDetection
# from models.sam_localization import SAMLocalisationModel
from utils.utils import compute_batch_iou, compute_batch_localization_f1, compute_batch_ap, generate_outputs, find_best_threshold, compute_accuracy_detection, compute_average_precision_detection

from sklearn.metrics import average_precision_score, accuracy_score


def validate_detection(model: CLIPModelDetection, data_loader: torch.utils.data.DataLoader, dataset_name: str = None):
    model.eval()
    all_img_paths = []
    all_predictions = []
    all_labels = []
    all_logits = []
    all_probabilities = []

    print('Length of dataset: ', (len(data_loader.dataset)))
    with torch.no_grad():
        for images, labels, img_names in tqdm(data_loader):
            images = images.to('cuda')
            logits = model(images)
            
            predictions = torch.argmax(logits, dim=1) # for this type of task we will switch to cross-entropy loss 
                                                      # since we might want to do classification as well
            
            probabilities = torch.softmax(logits, dim=1)
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_img_paths.extend(img_names)
            all_logits.append(logits)
            all_probabilities.append(probabilities)
    
    all_logits = torch.cat(all_logits, dim=0)  # [N, num_classes]
    all_probabilities = torch.cat(all_probabilities, dim=0)
    all_labels_tensor = torch.tensor(all_labels) # [N]
    
    # logits_np = all_logits.detach().cpu().numpy()
    logits_np = torch.softmax(all_logits, dim=1).detach().cpu().numpy()
    labels_np = all_labels_tensor.detach().cpu().numpy()
    probabilities_np = all_probabilities.detach().cpu().numpy()
    
    num_classes = logits_np.shape[1]
    labels_np = labels_np.astype(int)
    labels_one_hot = np.eye(num_classes)[labels_np]
    
    if dataset_name is None:
        # ap = average_precision_score(labels_one_hot, logits_np, average='macro')
        ap = average_precision_score(labels_one_hot, probabilities_np, average='macro')
    else: # for a specific test dataset, in a classification task, the ap will be computed wrongly due to a lack of data in the other classes
        dataset_label = {
            'real': 0,
            'ldm': 1,
            'repaint': 2,
            'lama': 3,
            'pluralistic': 4,
        }
        for key in dataset_label:
            if key in dataset_name:
                ap = average_precision_score(
                    labels_one_hot[:, [dataset_label['real'], dataset_label[key]]], 
                    logits_np[:, [dataset_label['real'], dataset_label[key]]], 
                    average='macro')
                break
    acc = accuracy_score(labels_np, [p.item() for p in all_predictions])
    
    return ap, acc, all_img_paths


def label_comparison(pred, label, comparison_type):
    # pred should contain probabilities, after applying sigmoid
    B, S = pred.shape
    H = int(S**0.5)
    W = int(S**0.5)
    pred = pred.view(B, H, W)
    
    if comparison_type == 'expansion': # expand label on the whole image
        label = label.view(B, 1, 1).expand(B, H, W)
        return pred, label # here BCEWithLogitsLoss can be applied, just like in the fully-supervised scenario
        
    elif comparison_type == 'max_pooling':
        pred = pred.view(B, -1) # flatten back
        pred = pred.max(dim=1).values # extract max pixel probability values
        return pred, label # here we can use BCELoss (unlike fully-supervised, we applied sigmoid; unlike detection, we use single channel probability, so no CrossEntropy)
        
    elif comparison_type == 'avg_pooling':
        pred_mean = pred.mean(dim=(1, 2)) # (B,)
        pred_mean = pred_mean.clamp(min=1e-6) # against log(0)
        pred = torch.log(pred_mean) + 0.7 # mean should be around 0.5 and we need to get the value to positives
        return pred, label # here we can use BCELoss (unlike fully-supervised, we applied sigmoid; unlike detection, we use single channel probability, so no CrossEntropy)

def validate_weakly_supervised_localization(model: CLIPModelLocalisation, data_loader: torch.utils.data.DataLoader, label_comparison_type: str, dataset_name: str, output_save_path: str = ''):
    model.eval()
    all_img_paths = []
    all_predictions = []
    all_labels = []
    all_scores = []

    print('Length of dataset: ', (len(data_loader.dataset)))
    with torch.no_grad():
        for images, labels, img_names in tqdm(data_loader):
            images = images.to('cuda')
            predictions = torch.sigmoid(model(images)).squeeze(1)
            
            processed_preds, processed_labels = label_comparison(predictions, labels, label_comparison_type)
            
            processed_preds_bin = (processed_preds > 0.5).float()
            
            if label_comparison_type == 'expansion':
                # comparing masks
                # since we don't have access to masks, we can only monitor the loss
                scores = torch.nn.functional.binary_cross_entropy(processed_preds_bin, processed_labels.to('cuda').float(), reduction='none').mean(dim=(1, 2))
                
            else:
                # comparing label numbers
                scores = processed_preds_bin.detach().cpu()
                
            # back to mask dimensions
            predictions = predictions.view(predictions.size(0), int(predictions.size(1)**0.5), int(predictions.size(1)**0.5))
            resized_predictions = []
            for i, pred in enumerate(predictions):
                if pred.size() != torch.Size([256, 256]):
                    pred_resized = F.resize(pred.unsqueeze(0), torch.Size([256, 256]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).squeeze(0)
                    resized_predictions.append(pred_resized)
                else:
                    resized_predictions.append(pred)

            # save predicted masks
            if output_save_path:
                generate_outputs(output_save_path + "/" + dataset_name, resized_predictions, img_names)
                
            all_predictions.append(processed_preds_bin.detach().cpu())
            all_labels.append(processed_labels.detach().cpu())
            all_img_paths.extend(img_names)
            all_scores.append(scores)
            
    all_predictions = torch.cat([x.reshape(-1) for x in all_predictions])
    all_labels = torch.cat([x.reshape(-1) for x in all_labels])
    all_scores = torch.cat([x.reshape(-1) for x in all_scores])

    if label_comparison_type != 'expansion':
        ap = average_precision_score(all_labels.numpy(), all_scores.numpy())
        acc = accuracy_score(all_labels.numpy(), all_predictions.numpy())
        
        return ap, acc, all_img_paths
    else:
        return None, None, all_img_paths
    
def validate_fully_supervised_localization(model, #: Union[CLIPModelLocalisation, SAMLocalisationModel], 
data_loader: torch.utils.data.DataLoader, dataset_name: str, output_save_path: str = ''):
    model.eval()
    all_img_paths = []
    all_ious = []
    all_f1_best = []
    all_f1_fixed = []
    all_mean_ap = []

    print('Length of dataset: ', (len(data_loader.dataset)))
    with torch.no_grad():
        for images, _, img_names, masks_paths in tqdm(data_loader):
            images = images.to('cuda')
            predictions = torch.sigmoid(model(images)).squeeze(1)

            # mask processing for the datasets
            if dataset_name in ["ldm", "lama", "pluralistic", "repaint-p2-9k"] or 'train' in dataset_name:
                masks = [ImageOps.invert(Image.open(mask_path).convert("L")) for mask_path in masks_paths]
            else:
                masks = [Image.open(mask_path).convert("L") for mask_path in masks_paths]

            masks = [ ((transforms.ToTensor()(mask).to(predictions.device)) > 0.5).float().squeeze() for mask in masks]

            # back to mask dimensions
            predictions = predictions.view(predictions.size(0), int(predictions.size(1)**0.5), int(predictions.size(1)**0.5))
            resized_predictions = []
            for i, pred in enumerate(predictions):
                if pred.size() != masks[i].size():
                    pred_resized = F.resize(pred.unsqueeze(0), masks[i].size(), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).squeeze(0)
                    resized_predictions.append(pred_resized)
                else:
                    resized_predictions.append(pred)

            batch_ious = compute_batch_iou(resized_predictions, masks, threshold=0.5)
            batch_f1_best, batch_f1_fixed = compute_batch_localization_f1(resized_predictions, masks)
            batch_ap = compute_batch_ap(resized_predictions, masks)

            # save predicted masks
            if output_save_path:
                generate_outputs(output_save_path + "/" + dataset_name, resized_predictions, img_names)

            all_ious.extend(batch_ious)
            all_f1_best.extend(batch_f1_best)
            all_f1_fixed.extend(batch_f1_fixed)
            all_img_paths.extend(img_names)
            all_mean_ap.extend(batch_ap)

    return all_ious, all_f1_best, all_f1_fixed, all_mean_ap, all_img_paths

    
def validate_fully_supervised_localization_real_images(model: CLIPModelLocalisation, data_loader: torch.utils.data.DataLoader, dataset_name: str, output_save_path: str = ''):
    model.eval()
    all_img_paths = []
    
    print('Length of dataset: ', (len(data_loader.dataset)))
    with torch.no_grad():
        for images, img_names in tqdm(data_loader):
            images = images.to('cuda')
            predictions = torch.sigmoid(model(images)).squeeze(1)

            # back to mask dimensions
            predictions = predictions.view(predictions.size(0), int(predictions.size(1)**0.5), int(predictions.size(1)**0.5))
            resized_predictions = []
            for i, pred in enumerate(predictions):
                if pred.size() != torch.Size([256, 256]):
                    pred_resized = F.resize(pred.unsqueeze(0), torch.Size([256, 256]), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).squeeze(0)
                    resized_predictions.append(pred_resized)
                else:
                    resized_predictions.append(pred)

            # save predicted masks
            if output_save_path:
                generate_outputs(output_save_path + "/" + dataset_name, resized_predictions, img_names)

            all_img_paths.extend(img_names)

    print('Len resized pred', len(resized_predictions))

    return all_img_paths


def validate_ensemble_fully_supervised_localization(ensemble, data_loader: torch.utils.data.DataLoader, dataset_name: str, output_save_path: str = ''):
    all_img_paths = []
    all_ious = []
    all_f1_best = []
    all_f1_fixed = []
    all_mean_ap = []

    print('Length of dataset: ', (len(data_loader.dataset)))
    with torch.no_grad():
        for images, _, img_names, masks_paths in tqdm(data_loader):
            # mask processing for the datasets
            if dataset_name in ["ldm", "lama", "pluralistic", "repaint-p2-9k"] or 'train' in dataset_name:
                masks = [
                    ImageOps.invert(Image.open(mask_path).convert("L")) if 'real_image' not in mask_path else torch.zeros((256, 256))
                    for mask_path in masks_paths
                ]
            else:
                masks = [Image.open(mask_path).convert("L") for mask_path in masks_paths]
            
            masks = [((mask if isinstance(mask, torch.Tensor) else transforms.ToTensor()(mask)) > 0.5).float().squeeze() for mask in masks]

            if ensemble.predict.__code__.co_argcount == 3:
                resized_predictions = ensemble.predict(images, masks)
            else:
                resized_predictions = ensemble.predict(images)
            
            batch_ious = compute_batch_iou(resized_predictions, masks, threshold=0.5)
            batch_f1_best, batch_f1_fixed = compute_batch_localization_f1(resized_predictions, masks)
            batch_ap = compute_batch_ap(resized_predictions, masks)

            # save predicted masks
            if output_save_path:
                generate_outputs(output_save_path + "/" + dataset_name, resized_predictions, img_names)

            all_ious.extend(batch_ious)
            all_f1_best.extend(batch_f1_best)
            all_f1_fixed.extend(batch_f1_fixed)
            all_img_paths.extend(img_names)
            all_mean_ap.extend(batch_ap)

    return all_ious, all_f1_best, all_f1_fixed, all_mean_ap, all_img_paths