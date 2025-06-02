import os
import pickle
from random import shuffle
from PIL import Image, ImageOps, ImageFile
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import List, Tuple
import torch

from utils.utils import get_images_from_path
from parameters import Parameters

# constants for mean and std - used in CLIP
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


class DeepfakeLocalizationDataset(Dataset):
    def __init__(self, params: Parameters) -> None:
        self.params = params

        # match CLIP input size and normalize
        self.image_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
        if 'SAM' in self.params.arch:
            self.image_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]) # normalizing causes trouble with SAM processing
        # match output size of the model
        self.mask_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

        # set images and masks paths
        if self.params.data_label == "train":
            self.fake_images_path = self.params.train_fake_path
            self.real_images_path = self.params.train_real_path
            self.masks_path = self.params.train_masks_ground_truth_path
        elif self.params.data_label == "valid":
            self.fake_images_path = self.params.valid_fake_path
            self.real_images_path = self.params.valid_real_path
            self.masks_path = self.params.valid_masks_ground_truth_path
        elif self.params.data_label == "test":
            self.fake_images_path = self.params.test_fake_path
            self.real_images_path = self.params.test_real_path
            self.masks_path = self.params.test_masks_ground_truth_path

        # get images
        self.fake_images = get_images_from_path(self.fake_images_path)
        shuffle(self.fake_images)
        # set the mask labels paths - they should be .png
        self.image_labels = {img: os.path.basename(img).replace('.jpg', '.png') for img in self.fake_images}
        
    def get_mask(self, mask_file_name):
        mask = Image.open(os.path.join(self.masks_path, mask_file_name)).convert("L") # binary mask
        if self.params.train_dataset in ['lama', 'ldm', 'repaint-p2-9k', 'pluralistic'] or 'train' in self.params.train_dataset:
            mask = ImageOps.invert(mask) # invert the mask if needed
        mask = self.mask_transforms(mask).view(-1)
        return mask
    
    def __len__(self):
        return len(self.fake_images)
    
    def __getitem__(self, index: int) -> Tuple[Image.Image, torch.Tensor, str, str]:
        img_path = self.fake_images[index]
        label = self.get_mask(self.image_labels[img_path])
        img = self.image_transforms(Image.open(img_path).convert("RGB"))
        return img, label, img_path, os.path.join(self.masks_path, self.image_labels[img_path])


class DeepfakeLocalizationRealImagesDataset(Dataset):
    def __init__(self, params: Parameters) -> None:
        self.params = params

        # match CLIP input size and normalize
        self.image_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
        # match output size of the model
        self.mask_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

        # set images and masks paths
        if self.params.data_label == "train":
            self.fake_images_path = self.params.train_fake_path
            self.real_images_path = self.params.train_real_path
            self.masks_path = self.params.train_masks_ground_truth_path
        elif self.params.data_label == "valid":
            self.fake_images_path = self.params.valid_fake_path
            self.real_images_path = self.params.valid_real_path
            self.masks_path = self.params.valid_masks_ground_truth_path
        elif self.params.data_label == "test":
            self.fake_images_path = self.params.test_fake_path
            self.real_images_path = self.params.test_real_path
            self.masks_path = self.params.test_masks_ground_truth_path

        # get images
        self.fake_images = get_images_from_path(self.fake_images_path)
        self.real_images = get_images_from_path(self.real_images_path)
        
    def get_mask(self, mask_file_name):
        mask = Image.open(os.path.join(self.masks_path, mask_file_name)).convert("L") # binary mask
        if self.params.train_dataset in ['lama', 'ldm', 'repaint-p2-9k', 'pluralistic'] or 'train' in self.params.train_dataset:
            mask = ImageOps.invert(mask) # invert the mask if needed
        mask = self.mask_transforms(mask).view(-1)
        return mask
    
    def __len__(self):
        return len(self.fake_images)
    
    def __getitem__(self, index: int) -> Tuple[Image.Image, torch.Tensor, str, str]:
        img_path = self.real_images[index]
        img = self.image_transforms(Image.open(img_path).convert("RGB"))
        return img, img_path



class DeepfakeDetectionDataset(Dataset):
    def __init__(self, params: Parameters) -> None:
        self.params = params

        # match CLIP input size and normalize
        self.image_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
        # match output size of the model
        self.mask_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

        # set images and masks paths
        if self.params.data_label == "train":
            self.fake_images_path = self.params.train_fake_path
            self.real_images_path = self.params.train_real_path
            self.masks_path = self.params.train_masks_ground_truth_path
        elif self.params.data_label == "valid":
            self.fake_images_path = self.params.valid_fake_path
            self.real_images_path = self.params.valid_real_path
            self.masks_path = self.params.valid_masks_ground_truth_path
        elif self.params.data_label == "test":
            self.fake_images_path = self.params.test_fake_path
            self.real_images_path = self.params.test_real_path
            self.masks_path = self.params.test_masks_ground_truth_path

        # get images
        self.fake_images = get_images_from_path(self.fake_images_path)
        self.real_images = get_images_from_path(self.real_images_path)
        self.all_images = self.fake_images + self.real_images
        shuffle(self.all_images)
        # set labels - 0 for real, 1 for fake
        self.image_labels = {img: 0 for img in self.real_images}
        self.image_labels.update({img: 1 for img in self.fake_images})

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index: int) -> Tuple[Image.Image, int, str]:
        img_path = self.all_images[index]
        img = self.image_transforms(Image.open(img_path).convert("RGB"))
        return img, self.image_labels[img_path], img_path


class DeepfakeClassificationDataset(Dataset):
    def __init__(self, params: Parameters) -> None:
        self.params = params

        # match CLIP input size and normalize
        self.image_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
        # match output size of the model
        self.mask_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

        # set images and masks paths
        if self.params.data_label == "train":
            self.fake_images_path = self.params.train_fake_path
            self.real_images_path = self.params.train_real_path
            self.masks_path = self.params.train_masks_ground_truth_path
        elif self.params.data_label == "valid":
            self.fake_images_path = self.params.valid_fake_path
            self.real_images_path = self.params.valid_real_path
            self.masks_path = self.params.valid_masks_ground_truth_path
        elif self.params.data_label == "test":
            self.fake_images_path = self.params.test_fake_path
            self.real_images_path = self.params.test_real_path
            self.masks_path = self.params.test_masks_ground_truth_path

        # get images
        self.fake_images = get_images_from_path(self.fake_images_path)
        self.real_images = get_images_from_path(self.real_images_path)
        self.all_images = self.fake_images + self.real_images
        shuffle(self.all_images)
        # set labels - 0 for real, 1 - ldm, 2 - repaint, 3 - lama, 4 - pluralistic
        
        self.label_map = {
            'real': 0,
            'ldm': 1,
            'repaint': 2,
            'lama': 3,
            'pluralistic': 4
        }
        
        self.image_labels = {img: 0 for img in self.real_images}
        
        if 'train_all_4' in self.fake_images_path:
            self.image_labels.update({
                img: self.label_map[keyword]
                for img in self.fake_images
                for keyword in self.label_map
                if keyword in img
            })
        elif 'ldm' in self.fake_images_path:
            self.image_labels.update({img: 1 for img in self.fake_images})
        elif 'repaint' in self.fake_images_path:
            self.image_labels.update({img: 2 for img in self.fake_images})
        elif 'lama' in self.fake_images_path:
            self.image_labels.update({img: 3 for img in self.fake_images})
        elif 'pluralistic' in self.fake_images_path:
            self.image_labels.update({img: 4 for img in self.fake_images})

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index: int) -> Tuple[Image.Image, int, str]:
        img_path = self.all_images[index]
        img = self.image_transforms(Image.open(img_path).convert("RGB"))
        return img, self.image_labels[img_path], img_path


class DeepfakeLocalizationWithRealImagesDataset(Dataset):
    def __init__(self, params: Parameters) -> None:
        self.params = params

        # match CLIP input size and normalize
        self.image_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
        # match output size of the model
        self.mask_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

        # set images and masks paths
        if self.params.data_label == "train":
            self.fake_images_path = self.params.train_fake_path
            self.real_images_path = self.params.train_real_path
            self.masks_path = self.params.train_masks_ground_truth_path
        elif self.params.data_label == "valid":
            self.fake_images_path = self.params.valid_fake_path
            self.real_images_path = self.params.valid_real_path
            self.masks_path = self.params.valid_masks_ground_truth_path
        elif self.params.data_label == "test":
            self.fake_images_path = self.params.test_fake_path
            self.real_images_path = self.params.test_real_path
            self.masks_path = self.params.test_masks_ground_truth_path

        # it should be used only on train_all_4 combined dataset
        # get images
        self.fake_images = get_images_from_path(self.fake_images_path)
        self.real_images = get_images_from_path(self.real_images_path)
        self.all_images = self.fake_images + self.real_images
        shuffle(self.all_images)

        self.image_labels = {img: os.path.basename(img).replace('.jpg', '.png') for img in self.fake_images}
        self.image_labels.update({img: None for img in self.real_images})

    def get_mask(self, mask_file_name):
        if mask_file_name is None: # real image
            return torch.zeros((256, 256), dtype=torch.float32).view(-1)
        mask = Image.open(os.path.join(self.masks_path, mask_file_name)).convert("L") # binary mask
        if self.params.train_dataset in ['lama', 'ldm', 'repaint-p2-9k', 'pluralistic'] or 'train' in self.params.train_dataset:
            mask = ImageOps.invert(mask) # invert the mask if needed
        mask = self.mask_transforms(mask).view(-1)
        return mask
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index: int) -> Tuple[Image.Image, torch.Tensor, str, str]:
        img_path = self.all_images[index]
        label = self.get_mask(self.image_labels[img_path])
        img = self.image_transforms(Image.open(img_path).convert("RGB"))
        return img, label, img_path, os.path.join(self.masks_path, self.image_labels[img_path]) if self.image_labels[img_path] is not None else 'real_image'



def get_dataloader(params: Parameters) -> DataLoader:
    if params.task_type == 'fully_supervised_localization':
        dataset = DeepfakeLocalizationDataset(params)
    elif params.task_type == 'detection':
        dataset = DeepfakeDetectionDataset(params)
    elif params.task_type == 'classification':
        dataset = DeepfakeClassificationDataset(params)
    elif params.task_type == 'fully_supervised_localization_real_images':
        dataset = DeepfakeLocalizationRealImagesDataset(params)
    elif params.task_type == 'fully_supervised_localization_with_real_images':
        dataset = DeepfakeLocalizationWithRealImagesDataset(params)
    elif params.task_type == 'weakly_supervised_localization': # here, only image-level labels are provided, having both real and fake images
        dataset = DeepfakeDetectionDataset(params) # therefore, it's the same setup as in detection, while keeping the conv decoder

    data_loader = DataLoader(dataset,
                             batch_size=params.batch_size,
                             shuffle=True if params.data_label == 'train' else False,
                             num_workers=params.num_threads,
                             pin_memory=torch.cuda.is_available(),
                             persistent_workers=params.num_threads > 0)
    return data_loader