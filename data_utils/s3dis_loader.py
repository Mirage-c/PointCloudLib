from jittor.dataset.dataset import Dataset
from pathlib import Path
import numpy as np
import time, pickle, glob, os
from os.path import join
from helper_ply import read_ply
from helper_tool import DataProcessing as DP

BASE_DIR = Path(__file__).parent
DATA_DIR = os.path.join(BASE_DIR, 'data/S3DIS')

class ConfigS3DIS:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 13  # Number of valid classes
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

cfg = ConfigS3DIS

class S3DIS(Dataset):
    def __init__(self, test_area_idx, partition='training'):
        self.name = 'S3DIS'
        self.path = DATA_DIR
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))
        # print("all_files: ", self.all_files.__str__())

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)
        self.partition = partition
        if partition == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif partition == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        
        self.possibility[self.partition] = []
        self.min_possibility[self.partition] = []
        # Random initialize
        for i, tree in enumerate(self.input_colors[self.partition]):
            self.possibility[self.partition] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[self.partition] += [float(np.min(self.possibility[self.partition][-1]))]

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def __getitem__(self, index):
        # Choose the cloud with the lowest probability
        cloud_idx = int(np.argmin(self.min_possibility[self.partition]))

        # choose the point with the minimum of possibility in the cloud as query point
        point_ind = np.argmin(self.possibility[self.partition][cloud_idx])

        # Get all points within the cloud from tree structure
        points = np.array(self.input_trees[self.partition][cloud_idx].data, copy=False)

        # Center point of input region
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to the center point
        noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)

        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < cfg.num_points:
            # Query all points within the cloud
            queried_idx = self.input_trees[self.partition][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.input_trees[self.partition][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

        # Shuffle index
        queried_idx = DP.shuffle_idx(queried_idx)
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_colors = self.input_colors[self.partition][cloud_idx][queried_idx]
        queried_pc_labels = self.input_labels[self.partition][cloud_idx][queried_idx]

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[self.partition][cloud_idx][queried_idx] += delta
        self.min_possibility[self.partition][cloud_idx] = float(np.min(self.possibility[self.partition][cloud_idx]))

        # up_sampled with replacement
        if len(points) < cfg.num_points:
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

        return (queried_pc_xyz.astype(np.float32),
                    queried_pc_colors.astype(np.float32),
                    queried_pc_labels,
                    queried_idx.astype(np.int32),
                    np.array([cloud_idx], dtype=np.int32))

if __name__ == "__main__":
    train = S3DIS(1)
    xyz, color, labels, idx, cloud = train[0]
    print(xyz.shape)
    print(color.shape)
    print(labels.shape)
    print(idx.shape)
    print(cloud.shape)
    