import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import os
import fcn
###############################################################################
# Functions
###############################################################################


def weight_init_googlenet(key, module, weights=None):

    if key == "LSTM":
        for name, param in module.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.xavier_normal_(param)
    elif weights is None:
        init.constant_(module.bias.data, 0.0)
        if key == "XYZ":
            init.normal_(module.weight.data, 0.0, 0.5)
        elif key == "LSTM":
            init.xavier_normal_(module.weight.data)
        else:
            init.normal_(module.weight.data, 0.0, 0.01)
    else:
        # print(key, weights[(key+"_1").encode()].shape, module.bias.size())
        module.bias.data[...] = torch.from_numpy(weights[(key+"_1").encode()])
        module.weight.data[...] = torch.from_numpy(weights[(key+"_0").encode()])
    return module

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def define_network(input_nc, lstm_hidden_size, model, init_from=None, isTest=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    if model == 'posenet':
        netG = PoseNet(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids)
    elif model == 'poselstm':
        netG = PoseLSTM(input_nc, lstm_hidden_size, weights=init_from, isTest=isTest, gpu_ids=gpu_ids)
    elif model == 'fcnlstm':
        netG = FCNLSTM(input_nc, lstm_hidden_size, weights=init_from, isTest=isTest, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % model)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()
##############################################################################
# Classes
##############################################################################

# defines the regression heads for googlenet
class RegressionHead(nn.Module):
    def __init__(self, lossID, weights=None, lstm_hidden_size=None):
        super(RegressionHead, self).__init__()
        self.has_lstm = lstm_hidden_size != None
        dropout_rate = 0.5 if lossID == "loss3" else 0.7
        nc_loss = {"loss1": 512, "loss2": 528}
        nc_cls = [1024, 2048] if lstm_hidden_size is None else [lstm_hidden_size*4, lstm_hidden_size*4]

        self.dropout = nn.Dropout(p=dropout_rate)
        if lossID != "loss3":
            self.projection = nn.Sequential(*[nn.AvgPool2d(kernel_size=5, stride=3),
                                              weight_init_googlenet(lossID+"/conv", nn.Conv2d(nc_loss[lossID], 128, kernel_size=1), weights),
                                              nn.ReLU(inplace=True)])
            self.cls_fc_pose = nn.Sequential(*[weight_init_googlenet(lossID+"/fc", nn.Linear(2048, 1024), weights),
                                               nn.ReLU(inplace=True)])
            self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(nc_cls[0], 3))
            self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(nc_cls[0], 4))
            if lstm_hidden_size is not None:
                self.lstm_pose_lr = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
                self.lstm_pose_ud = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
        else:
            self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
            self.cls_fc_pose = nn.Sequential(*[weight_init_googlenet("pose", nn.Linear(1024, 2048)),
                                               nn.ReLU(inplace=True)])
            self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(nc_cls[1], 3))
            self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(nc_cls[1], 4))

            if lstm_hidden_size is not None:
                self.lstm_pose_lr = weight_init_googlenet("LSTM", nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
                self.lstm_pose_ud = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))

    def forward(self, input):
        output = self.projection(input)
        output = self.cls_fc_pose(output.view(output.size(0), -1))
        if self.has_lstm:
            output = output.view(output.size(0),32, -1)
            _, (hidden_state_lr, _) = self.lstm_pose_lr(output.permute(0,1,2))
            _, (hidden_state_ud, _) = self.lstm_pose_ud(output.permute(0,2,1))
            output = torch.cat((hidden_state_lr[0,:,:],
                                hidden_state_lr[1,:,:],
                                hidden_state_ud[0,:,:],
                                hidden_state_ud[1,:,:]), 1)
        output = self.dropout(output)
        output_xy = self.cls_fc_xy(output)
        output_wpqr = self.cls_fc_wpqr(output)
        output_wpqr = F.normalize(output_wpqr, p=2, dim=1)
        return [output_xy, output_wpqr]

class RegressionHead_FCN(nn.Module):
    def __init__(self, lossID, weights=None, lstm_hidden_size=None):
        super(RegressionHead_FCN, self).__init__()
        self.has_lstm = lstm_hidden_size != None
        dropout_rate = 0.5 if lossID == "loss3" else 0.7
        nc_loss = {"loss1/conv": 512, "inception_3b/1x1": 256} # {"loss1": 512, "loss2": 528}
        num_fc_features = {"inception_3b/1x1": 41472, "loss1/conv": 10368} # {"loss1": 46208, "loss2": 10368}
        nc_cls = [1024, 2048] if lstm_hidden_size is None else [lstm_hidden_size*4, lstm_hidden_size*4]
        # key = {"loss1": "inception_3b/1x1", "loss2": "loss2/conv"}
        self.dropout = nn.Dropout(p=dropout_rate)
        if lossID != "loss3/conv": # "inception_3b/1x1", "loss1"
            self.projection = nn.Sequential(*[nn.AvgPool2d(kernel_size=5, stride=3),
                                              weight_init_googlenet(lossID, nn.Conv2d(nc_loss[lossID], 128, kernel_size=1), weights),
                                              # weight_init_googlenet(lossID+"/conv", nn.Conv2d(nc_loss[lossID], 128, kernel_size=1), weights),
                                              nn.ReLU(inplace=True)])
            self.cls_fc_pose = nn.Sequential(*[
                                            #  weight_init_googlenet(lossID+"/fc", nn.Linear(2048, 1024), weights),
                                               weight_init_googlenet('LSTM',nn.Linear(num_fc_features[lossID],1024)),
                                               nn.ReLU(inplace=True)])
            self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(nc_cls[0], 3))
            self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(nc_cls[0], 4))
            if lstm_hidden_size is not None:
                self.lstm_pose_lr = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
                self.lstm_pose_ud = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
        else: # "loss3"
            self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
            self.cls_fc_pose = nn.Sequential(*[
                                              #  weight_init_googlenet("pose", nn.Linear(1024, 2048)),
                                               weight_init_googlenet("pose", nn.Linear(27216, 2048)),
                                               nn.ReLU(inplace=True)])
            self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(nc_cls[1], 3))
            self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(nc_cls[1], 4))

            if lstm_hidden_size is not None:
                self.lstm_pose_lr = weight_init_googlenet("LSTM", nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
                self.lstm_pose_ud = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))

    def forward(self, input):
        output = self.projection(input)
        output = self.cls_fc_pose(output.view(output.size(0), -1))
        if self.has_lstm:
            output = output.view(output.size(0),32, -1)
            _, (hidden_state_lr, _) = self.lstm_pose_lr(output.permute(0,1,2))
            _, (hidden_state_ud, _) = self.lstm_pose_ud(output.permute(0,2,1))
            output = torch.cat((hidden_state_lr[0,:,:],
                                hidden_state_lr[1,:,:],
                                hidden_state_ud[0,:,:],
                                hidden_state_ud[1,:,:]), 1)
        output = self.dropout(output)
        output_xy = self.cls_fc_xy(output)
        output_wpqr = self.cls_fc_wpqr(output)
        output_wpqr = F.normalize(output_wpqr, p=2, dim=1)
        return [output_xy, output_wpqr]

# define inception block for GoogleNet
class InceptionBlock(nn.Module):
    def __init__(self, incp, input_nc, x1_nc, x3_reduce_nc, x3_nc, x5_reduce_nc,
                 x5_nc, proj_nc, weights=None, gpu_ids=[]):
        super(InceptionBlock, self).__init__()
        self.gpu_ids = gpu_ids
        # first
        self.branch_x1 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/1x1", nn.Conv2d(input_nc, x1_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True)])

        self.branch_x3 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/3x3_reduce", nn.Conv2d(input_nc, x3_reduce_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            weight_init_googlenet("inception_"+incp+"/3x3", nn.Conv2d(x3_reduce_nc, x3_nc, kernel_size=3, padding=1), weights),
            nn.ReLU(inplace=True)])

        self.branch_x5 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/5x5_reduce", nn.Conv2d(input_nc, x5_reduce_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            weight_init_googlenet("inception_"+incp+"/5x5", nn.Conv2d(x5_reduce_nc, x5_nc, kernel_size=5, padding=2), weights),
            nn.ReLU(inplace=True)])

        self.branch_proj = nn.Sequential(*[
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            weight_init_googlenet("inception_"+incp+"/pool_proj", nn.Conv2d(input_nc, proj_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True)])

        if incp in ["3b", "4e"]:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pool = None

    def forward(self, input):
        outputs = [self.branch_x1(input), self.branch_x3(input),
                   self.branch_x5(input), self.branch_proj(input)]
        # print([[o.size()] for o in outputs])
        output = torch.cat(outputs, 1)
        if self.pool is not None:
            return self.pool(output)
        return output

class PoseNet(nn.Module):
    def __init__(self, input_nc, weights=None, isTest=False,  gpu_ids=[]):
        super(PoseNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        self.before_inception = nn.Sequential(*[
            weight_init_googlenet("conv1/7x7_s2", nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            weight_init_googlenet("conv2/3x3_reduce", nn.Conv2d(64, 64, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            weight_init_googlenet("conv2/3x3", nn.Conv2d(64, 192, kernel_size=3, padding=1), weights),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ])

        self.inception_3a = InceptionBlock("3a", 192, 64, 96, 128, 16, 32, 32, weights, gpu_ids)
        self.inception_3b = InceptionBlock("3b", 256, 128, 128, 192, 32, 96, 64, weights, gpu_ids)
        self.inception_4a = InceptionBlock("4a", 480, 192, 96, 208, 16, 48, 64, weights, gpu_ids)
        self.inception_4b = InceptionBlock("4b", 512, 160, 112, 224, 24, 64, 64, weights, gpu_ids)
        self.inception_4c = InceptionBlock("4c", 512, 128, 128, 256, 24, 64, 64, weights, gpu_ids)
        self.inception_4d = InceptionBlock("4d", 512, 112, 144, 288, 32, 64, 64, weights, gpu_ids)
        self.inception_4e = InceptionBlock("4e", 528, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
        self.inception_5a = InceptionBlock("5a", 832, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
        self.inception_5b = InceptionBlock("5b", 832, 384, 192, 384, 48, 128, 128, weights, gpu_ids)

        self.cls1_fc = RegressionHead(lossID="loss1", weights=weights)
        self.cls2_fc = RegressionHead(lossID="loss2", weights=weights)
        self.cls3_fc = RegressionHead(lossID="loss3", weights=weights)

        self.model = nn.Sequential(*[self.inception_3a, self.inception_3b,
                                   self.inception_4a, self.inception_4b,
                                   self.inception_4c, self.inception_4d,
                                   self.inception_4e, self.inception_5a,
                                   self.inception_5b, self.cls1_fc,
                                   self.cls2_fc, self.cls3_fc
                                   ])
        if self.isTest:
            self.model.eval() # ensure Dropout is deactivated during test

    def forward(self, input):

        output_bf = self.before_inception(input)
        output_3a = self.inception_3a(output_bf)
        output_3b = self.inception_3b(output_3a)
        output_4a = self.inception_4a(output_3b)
        output_4b = self.inception_4b(output_4a)
        output_4c = self.inception_4c(output_4b)
        output_4d = self.inception_4d(output_4c)
        output_4e = self.inception_4e(output_4d)
        output_5a = self.inception_5a(output_4e)
        output_5b = self.inception_5b(output_5a)

        if not self.isTest:
            return self.cls1_fc(output_4a) + self.cls2_fc(output_4d) +  self.cls3_fc(output_5b)
        return self.cls3_fc(output_5b)

class PoseLSTM(PoseNet):
    def __init__(self, input_nc, lstm_hidden_size, weights=None, isTest=False,  gpu_ids=[]):
            super(PoseLSTM, self).__init__(input_nc, weights, isTest, gpu_ids)
            self.cls1_fc = RegressionHead(lossID="loss1", weights=weights, lstm_hidden_size=lstm_hidden_size)
            self.cls2_fc = RegressionHead(lossID="loss2", weights=weights, lstm_hidden_size=lstm_hidden_size)
            self.cls3_fc = RegressionHead(lossID="loss3", weights=weights, lstm_hidden_size=lstm_hidden_size)

            self.model = nn.Sequential(*[self.inception_3a, self.inception_3b,
                                       self.inception_4a, self.inception_4b,
                                       self.inception_4c, self.inception_4d,
                                       self.inception_4e, self.inception_5a,
                                       self.inception_5b, self.cls1_fc,
                                       self.cls2_fc, self.cls3_fc
                                       ])
            if self.isTest:
                self.model.eval() # ensure Dropout is deactivated during test

class FCN16s(nn.Module):

    pretrained_model = \
        os.path.expanduser('~/data/models/pytorch/fcn16s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vVGE3TkRMbWlNRms',
            path=cls.pretrained_model,
            md5='991ea45d30d632a01e5ec48002cac617',
        )

    def __init__(self, n_class=21):
        super(FCN16s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(
            n_class, n_class, 32, stride=16, bias=False)

class FCNLSTM(nn.Module):

    pretrained_model = \
        os.path.expanduser('~/data/models/pytorch/fcn8s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU',
            path=cls.pretrained_model,
            md5='dbd9bbb3829a3184913bccc74373afbb',
        )

    def __init__(self, input_nc, lstm_hidden_size, weights=None, isTest=False,  gpu_ids=[], n_class=21):
        super(FCNLSTM, self).__init__()#input_nc, lstm_hidden_size, weights=None, isTest=False,  gpu_ids=[], n_class=21)
        self.n_class = n_class
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        self.weight_fcn16s = weights[1]

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, self.n_class, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_class, 1)
        self.score_pool4 = nn.Conv2d(512, self.n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            self.n_class, self.n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            self.n_class, self.n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            self.n_class, self.n_class, 4, stride=2, bias=False)

        self.cls1_fc = RegressionHead_FCN(lossID="inception_3b/1x1", weights=weights, lstm_hidden_size=lstm_hidden_size)
        self.cls2_fc = RegressionHead_FCN(lossID="loss1/conv", weights=weights, lstm_hidden_size=lstm_hidden_size)
        self.cls3_fc = RegressionHead_FCN(lossID="loss3/conv", weights=weights, lstm_hidden_size=lstm_hidden_size)

        if self.isTest:
            self.model.eval() # ensure Dropout is deactivated during test
        self._initialize_weights()

    def _initialize_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.zero_()
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     if isinstance(m, nn.ConvTranspose2d):
        #         assert m.kernel_size[0] == m.kernel_size[1]
        #         initial_weight = get_upsampling_weight(
        #             m.in_channels, m.out_channels, m.kernel_size[0])
        #         m.weight.data.copy_(initial_weight)
        for name, l2 in self.named_children():
            if hasattr(self.weight_fcn16s, name):
                # print('[debug]{}'.format(name), end='')
                l1 = getattr(self.weight_fcn16s, name)
                if hasattr(l1, 'weight'):  # '[Error]{} in the pretrained model does not have weight!'.format(name)
                    assert l1.weight.size() == l2.weight.size(), '[Error]The size of {}.weight does not match with the pretrained model!'.format(
                        name)
                    l2.weight.data.copy_(l1.weight.data)
                    # print(' loaded weight', end='')
                    if l1.bias is not None:
                        assert l1.bias.size() == l2.bias.size(), '[Error]The size of {}.bias does not match with the pretrained model!'.format(
                            name)
                        l2.bias.data.copy_(l1.bias.data)
                        # print(' and bias', end='')
                # print()
            else:
                print("[debug]{}, {}".format(name, type(l2)))
                if isinstance(l2, torch.nn.Conv2d) or isinstance(l2, torch.nn.ConvTranspose2d):
                    for para_name, param in l2.named_parameters():
                        if 'bias' in para_name:
                            torch.nn.init.constant_(param, 0.0)
                        elif 'weight' in para_name:
                            torch.nn.init.xavier_normal_(param)

    def forward(self, x):
        output1_1 = self.relu1_1(self.conv1_1(x))
        output1_2 = self.relu1_2(self.conv1_2(output1_1))
        pool1 = self.pool1(output1_2)

        output2_1 = self.relu2_1(self.conv2_1(pool1))
        output2_2 = self.relu2_2(self.conv2_2(output2_1))
        pool2 = self.pool2(output2_1)

        output3_1 = self.relu3_1(self.conv3_1(pool2))
        output3_2 = self.relu3_2(self.conv3_2(output3_1))
        output3_3 = self.relu3_3(self.conv3_3(output3_2))
        pool3 = self.pool3(output3_1)  # 1/8

        output4_1 = self.relu4_1(self.conv4_1(pool3))
        output4_2 = self.relu4_2(self.conv4_2(output4_1))
        output4_3 = self.relu4_3(self.conv4_3(output4_2))
        pool4 = self.pool4(output4_3)  # 1/16

        output5_1 = self.relu5_1(self.conv5_1(pool4))
        output5_2 = self.relu5_2(self.conv5_2(output5_1))
        output5_3 = self.relu5_3(self.conv5_3(output5_2))
        pool5 = self.pool5(output5_3)

        output6 = self.drop6(self.relu6(self.fc6(pool5)))

        output7 = self.drop7(self.relu7(self.fc7(output6)))

        score_fr = self.score_fr(output7)
        upscore2 = self.upscore2(score_fr)  # 1/16

        h = self.score_pool4(pool4)
        score_pool4c = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]  # 1/16

        upscore_pool4 = self.upscore_pool4(upscore2 + score_pool4c)  # 1/8

        h = self.score_pool3(pool3)
        score_pool3c = h[:, :,
                       9:9 + upscore_pool4.size()[2],
                       9:9 + upscore_pool4.size()[3]]  # 1/8

        h = upscore_pool4 + score_pool3c
        # h = self.upscore8(upscore_pool4 + score_pool3c)
        # h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        if self.isTest:
            output = self.cls3_fc(h)
        else:
            output = self.cls1_fc(pool3) + self.cls2_fc(pool4) + self.cls3_fc(h)
            # output = self.cls2_fc(pool4) + self.cls3_fc(h)

        return output

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)