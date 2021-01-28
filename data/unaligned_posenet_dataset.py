import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_posenet_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import numpy
from util.util import scale

class UnalignedPoseNetDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        split_file = os.path.join(self.root , 'dataset_'+opt.phase+'.txt')
        self.A_paths = numpy.loadtxt(split_file, dtype=str, delimiter=' ', skiprows=3, usecols=(0))
        self.A_paths = [os.path.join(self.root, path) for path in self.A_paths]
        self.A_poses = numpy.loadtxt(split_file, dtype=float, delimiter=' ', skiprows=3, usecols=(1,2,3,4,5,6,7))
        if opt.isTrain:
            # scale values of location to defined range
            self.A_poses[:, :3], opt.position_range = scale(self.A_poses[:, :3], self.opt.scale_range)
            # TODO find a better way to store position_range
            file_name = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'opt_'+self.opt.phase+'.txt')
            with open(file_name, 'at') as opt_file:
                opt_file.write('position_range: {}, {}\n',format(opt.position_range))
        # else:
        #     read the position_range used for training from opt_train.txt
        #     path_train_file = os.path.join(opt.checkpoints_dir, opt.name, 'opt_train.txt')
        #     with open(path_train_file, 'rt') as ftrain:
        #         for line in ftrain:
        #             l = line.split(':')
                    # if 'position_range' == l[0]:



        if opt.model == "poselstm":
            self.mean_image = None
            print("mean image subtraction is deactivated")
        else:
            self.mean_image = numpy.load(os.path.join(self.root, 'mean_image.npy'))

        self.A_size = len(self.A_paths)
        self.transform = get_posenet_transform(opt, self.mean_image)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        # index_A = index % self.A_size
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        A_pose = self.A_poses[index % self.A_size]

        A = self.transform(A_img)

        return {'A': A, 'B': A_pose,
                'A_paths': A_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'UnalignedPoseNetDataset'
