import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import numpy
import datetime
import pytz
from tqdm import *

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

results_dir = os.path.join(opt.results_dir, opt.name)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print("Results directory %s created." % results_dir)

besterror  = [0, float('inf'), float('inf')] # nepoch, medX, medQ

if opt.which_epoch.isnumeric():
    epoch_start = int(opt.which_epoch)
    assert epoch_start > 0 and epoch_start <= 500
    if opt.model == 'posenet':
        testepochs = numpy.arange(epoch_start, 500+1, opt.load_epoch_freq)
    else:
        testepochs = numpy.arange(epoch_start, 500+1, opt.load_epoch_freq)
        # testepochs = numpy.arange(450, 1200+1, 5)
else:
    print('Non numeric input detected for --which_epoch, test the latest only')
    testepochs = ['latest']

testfile = open(os.path.join(results_dir, 'test_median.txt'), 'a')
testfile.write(datetime.datetime.now(pytz.timezone('Asia/Singapore')).strftime("%y%m%d-%H%M%S\n"))
testfile.write('epoch\tmedX\tmedQ\n')
testfile.write('==================\n')

model = create_model(opt)
visualizer = Visualizer(opt)

for testepoch in tqdm(testepochs):
    model.load_network(model.netG, 'G', testepoch)
    visualizer.change_log_path(testepoch)
    # test
    # err_pos = []
    # err_ori = []
    err = []
    print("epoch: "+ str(testepoch))
    for i, data in tqdm(enumerate(dataset)):
        # data includes:
        # (tensor)      'A': the image
        # (tensor)      'B': pose
        # (string)      'A_paths' path of the image file
        model.set_input(data)
        model.test()
        img_path = model.get_image_paths()[0]
        if "\\" in img_path:
            imgname = img_path.split("\\")[-1]
        elif '/' in img_path:
            imgname = img_path.split('/')[-1]
        else:
            imgname = img_path
        print('\t{:04d}/{:04d}: processing image {}'.format(i + 1, len(dataset), imgname))
        # print('\t%04d/%04d: processing image %s' % (i+1, len(dataset), img_path), end='\r')
        image_path = img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
        pose = model.get_current_pose()
        visualizer.save_estimated_pose(image_path, pose)
        err_p, err_o = model.get_current_errors()
        # err_pos.append(err_p)
        # err_ori.append(err_o)
        err.append([err_p, err_o])

    median_pos = numpy.median(err, axis=0)
    if median_pos[0] < besterror[1]:
        besterror = [testepoch, median_pos[0], median_pos[1]]
    print()
    # print("median position: {0:.2f}".format(numpy.median(err_pos)))
    # print("median orientat: {0:.2f}".format(numpy.median(err_ori)))
    print("\tmedian wrt pos.: {0:.2f}m {1:.2f}°".format(median_pos[0], median_pos[1]))
    testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(testepoch,
                                                     median_pos[0],
                                                     median_pos[1]))
    testfile.flush()
print("Best epoch:\nepoch\tmedX\tmedQ")
print("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('-----------------\n')
testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('==================\n')
testfile.close()
