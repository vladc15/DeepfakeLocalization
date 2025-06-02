from .datasets import *

# paths to dolos datasets
def get_test_dataset_paths(dataset):
    # general_dataset can be either celebahq or ffhq, but we will stick to celebahq
    # dataset can be lama, ldm, p2, pluralistic, repaint-p2, repaint-p2-9k
    paths = {
        'fake_path': f'datasets/dolos_data/celebahq/fake/{dataset}/images/test',
        'real_path': 'datasets/dolos_data/celebahq/real/test',
        'masks_path': f'datasets/dolos_data/celebahq/fake/{dataset}/masks/test',
        'key': dataset
    }
    return paths

LOCALIZATION_DATASET_PATHS = [
    get_test_dataset_paths('lama', 'celebahq'),
    get_test_dataset_paths('ldm', 'celebahq'),
    get_test_dataset_paths('pluralistic', 'celebahq'),
    get_test_dataset_paths('repaint-p2-9k', 'celebahq'),
]

DETECTION_DATASET_PATHS = [
    get_test_dataset_paths('lama', 'celebahq'),
    get_test_dataset_paths('ldm', 'celebahq'),
    get_test_dataset_paths('pluralistic', 'celebahq'),
    get_test_dataset_paths('repaint-p2-9k', 'celebahq'),
]