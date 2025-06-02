import os

class Parameters():
    def __init__(self):

        self.experiment_name = 'experiment_name'  # name of the experiment - where to store the obtained models
        self.task_type = 'fully_supervised_localization'  # fully supervised with local manipulation ground truth masks; other options could be 'detection' (or 'weakly_supervised_localization')  
        
        self.data_label = 'train' # train, valid or test

        self.arch = 'CLIP:ViT-L/14' # model architecture (using CLIP with ViT-L/14 backbone)
        self.fix_backbone = True # train only the decoder (and fc layers - the ones after the residual blocks, if existing) or the whole model
        
        self.weight_decay = 0.0 # weight for L2 regularization; leaving it to 0 will not enable L2 regularization (otherwise, set it around 1e-5)
        self.batch_size = 64 # input batch size
        self.num_threads = 4 # num threads for loading data
        self.init_type = 'normal' # network initialization (normal/xavier/kaiming/orthogonal)
        self.init_gain = 0.02 # scaling factor for normal, xavier and orthogonal initialization of layer weights

        # prepare dirs for saving the models and results
        # self.create_output_dirs()

        # train specific parameters
        self.train_dataset = 'pluralistic'
        self.decoder_type = 'conv-20' # options such as conv-4, conv-12, conv-20 for the decoder
        self.feature_layer = 'layer20' # layer of the backbone from which to extract features

        self.early_stop_epochs = 5 # number of epochs after which to stop training if no improvement is observed
        self.optim = 'adam'
        self.beta1 = 0.9 # momentum term of adam
        self.lr = 0.001 # initial lr for adam

        self.show_loss_freq = 50 # frequency of showing loss
        self.num_iter = 400 # total epochs

        # datasets paths
        self.data_root_path = None # for dolos, it is not needed, but for other datasets, it needs to be specified
        self.train_fake_path = 'datasets/dolos_data/celebahq/fake/ldm/images/train' # folder path to training fake data
        self.valid_fake_path = 'datasets/dolos_data/celebahq/fake/ldm/images/valid' # folder path to validation fake data
        self.test_fake_path = 'datasets/dolos_data/celebahq/fake/ldm/images/test' # folder path to test fake data
        self.train_masks_ground_truth_path = 'datasets/dolos_data/celebahq/fake/ldm/masks/train' # path to train ground truth masks (only for fully_supervised training)
        self.valid_masks_ground_truth_path = 'datasets/dolos_data/celebahq/fake/ldm/masks/valid' # path to validation ground truth masks (only for fully_supervised training)
        self.test_masks_ground_truth_path = 'datasets/dolos_data/celebahq/fake/ldm/masks/test' # path to test ground truth masks (only for fully_supervised training)
        self.train_real_path = 'datasets/dolos_data/celebahq/real/train' # folder path to training real data
        self.valid_real_path = 'datasets/dolos_data/celebahq/real/valid' # folder path to validation real data        
        self.test_real_path = 'datasets/dolos_data/celebahq/real/test' # folder path to test real data

        # test specific parameters
        self.checkpoint_path = '' # path to the trained model's checkpoint

    def create_output_dirs(self):
        self.save_dir = os.path.join('experiments', self.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_dir_models = os.path.join(self.save_dir, 'models')
        os.makedirs(self.save_dir_models, exist_ok=True)
        self.save_dir_results = os.path.join(self.save_dir, 'results')
        os.makedirs(self.save_dir_results, exist_ok=True)

    def update_dolos_data_paths(self, root_path: str, dataset_name: str = 'ldm'):
        assert root_path is not None, 'Root path for the dataset is not specified.'

        self.data_root_path = root_path.rstrip('/')
        self.train_dataset = dataset_name
        
        # obtain the paths for the dolos dataset - using the root path and the dataset name
        self.train_fake_path = os.path.join(self.data_root_path, 'fake', self.train_dataset, 'images', 'train')
        self.valid_fake_path = os.path.join(self.data_root_path, 'fake', self.train_dataset, 'images', 'valid')
        self.test_fake_path = os.path.join(self.data_root_path, 'fake', self.train_dataset, 'images', 'test')
        self.train_masks_ground_truth_path = os.path.join(self.data_root_path, 'fake', self.train_dataset, 'masks', 'train')
        self.valid_masks_ground_truth_path = os.path.join(self.data_root_path, 'fake', self.train_dataset, 'masks', 'valid')
        self.test_masks_ground_truth_path = os.path.join(self.data_root_path, 'fake', self.train_dataset, 'masks', 'test')
        self.train_real_path = os.path.join(self.data_root_path, 'real', 'train')
        self.valid_real_path = os.path.join(self.data_root_path, 'real', 'valid')
        self.test_real_path = os.path.join(self.data_root_path, 'real', 'test')