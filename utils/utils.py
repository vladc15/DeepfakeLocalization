import pickle
import os

# this file uses utilitary functions from utils.py of the DeCLIP repository
from utils.declip_utils import *

# datasets
def recursively_read(rootdir, exts=["png", "jpg", "JPEG", "jpeg", 'tif', 'tiff']):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if file.split('.')[-1] in exts:
                out.append(os.path.join(r, file))
    return out

def get_images_from_path(path):
    if path.endswith(".pickle"):
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        return image_list
    return recursively_read(path)

# defined after the same name function in utils.py, but with a fallback mechanism
def dynamic_threshold_metrics(preds, gt_dilated, gt_eroded):
    preds, gt_dilated, gt_eroded = preds.flatten(), gt_dilated.flatten(), gt_eroded.flatten()
    
    # fallback for empty predictions or ground truth (e.g. voting in ensemble)
    if preds.numel() == 0 or (gt_dilated + gt_eroded).sum() == 0:
        print("Empty predictions or ground truth, returning empty tensors in dynamic_threshold_metrics")
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    inds = torch.argsort(preds)
    inds = inds[(gt_dilated[inds] + gt_eroded[inds]) > 0]
    thresholds = preds[inds]
    gt_dilated, gt_eroded = gt_dilated[inds], gt_eroded[inds]
    tn = torch.cumsum(gt_dilated, dim=0)
    fn = torch.cumsum(gt_eroded, dim=0)
    fp, tp = torch.sum(gt_dilated) - tn, torch.sum(gt_eroded) - fn
    mask = F.pad(thresholds[1:] > thresholds[:-1], (0, 1), mode="constant")
    return fp[mask], tp[mask], fn[mask], tn[mask]

# defined after the same name function in utils.py, but with a fallback mechanism
def localization_f1(pred, gt, verbose=False):
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt, dtype=torch.float32)
    pred, gt = pred.float(), gt.float()
    gt_dilated, gt_eroded = extract_ground_truths(gt)

    # best threshold F1
    try:
        fp, tp, fn, tn = dynamic_threshold_metrics(pred, gt_dilated, gt_eroded)
        f1_dynamic = compute_f1(fp, tp, fn)
        if f1_dynamic.numel() == 0:
            if verbose:
                print("Empty predictions or ground truth, returning empty tensors in localization_f1")
            best_f1 = torch.tensor(0.0)
        else:
            best_f1 = torch.max(f1_dynamic)
    except Exception as e:
        print(e)
        best_f1 = torch.tensor(np.nan)

    # fixed threshold F1
    try:
        fp, tp, fn, tn = fixed_threshold_metrics(pred, gt, gt_dilated, gt_eroded, 0.5)
        f1_fixed = compute_f1(fp, tp, fn)
    except Exception as e:
        print(e)
        f1_fixed = torch.tensor(np.nan)

    return max(best_f1, f1_fixed), f1_fixed

# defined after the same name function in utils.py, but with a fallback mechanism
def compute_batch_localization_f1(preds_list, gts_list):
    assert len(preds_list) == len(gts_list), "Both lists must have the same length"
    
    batch_f1_scores_best = []
    batch_f1_scores_fixed = []
    for preds, gt in zip(preds_list, gts_list):
        best_f1, fixed_f1 = localization_f1(preds, gt)
        batch_f1_scores_best.append(best_f1.item())
        batch_f1_scores_fixed.append(fixed_f1.item())
    return batch_f1_scores_best, batch_f1_scores_fixed


def compute_mean_iou(ious, verbose=True, extra_text=''):
    mean_iou = sum(ious) / len(ious)
    if verbose:
        print(f"{extra_text}Mean IOU: {round(mean_iou, 2)}")
    return mean_iou

def compute_mean_f1(f1_best, f1_fixed, verbose=True, extra_text=''):
    mean_f1_best = sum(f1_best) / len(f1_best)
    mean_f1_fixed = sum(f1_fixed) / len(f1_fixed)
    
    if verbose:
        print(f"{extra_text}Mean F1 best: {round(mean_f1_best, 4)}")
        print(f"{extra_text}Mean F1 fixed: {round(mean_f1_fixed, 4)}")

    return mean_f1_best, mean_f1_fixed

def compute_mean_ap(mean_ap, verbose=True, extra_text=''):
    mean_ap = sum(mean_ap) / len(mean_ap)
    if verbose:
        print(f"{extra_text}Mean AP: {round(mean_ap, 4)}")
    return mean_ap

def compute_mean_acc_detection(logits, labels, verbose=True, extra_text=''):
    mean_acc = compute_accuracy_detection(logits, labels)
    if verbose:
        print(f"{extra_text}Mean ACC: {round(mean_acc, 2)}")
    return mean_acc

def compute_mean_ap_detection(logits, labels, verbose=True, extra_text=''):
    mean_ap = compute_average_precision_detection(logits, labels)
    if verbose:
        print(f"{extra_text}Mean AP: {round(mean_ap, 4)}")
    return mean_ap