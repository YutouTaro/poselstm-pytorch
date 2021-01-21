import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
# from tqdm.notebook import tqdm
from tqdm import *

opt = TrainOptions().parse()
## SEEDING
import torch
import numpy
import random
torch.manual_seed(opt.seed)
numpy.random.seed(opt.seed)
random.seed(opt.seed)
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
## SEEDING

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in tqdm(enumerate(dataset)):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        # if total_steps % opt.display_freq == 0:
        #     save_result = total_steps % opt.update_html_freq == 0
        #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        # if total_steps % opt.save_latest_freq == 0:
        #     print('saving the latest model (epoch %d, total_steps %d)' %
        #           (epoch, total_steps))
        #     model.save('latest')

    errors = model.get_current_errors()
    t = (time.time() - iter_start_time) / opt.batchSize
    visualizer.print_current_errors(epoch, epoch_iter, errors, t)
    msg_endOfEpoch = 'End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
    print(msg_endOfEpoch)
    with open(visualizer.log_name, "a") as log_file:
        log_file.write('%s\n' % msg_endOfEpoch)
    model.update_learning_rate()

    if epoch % opt.save_epoch_freq == 0 or (epoch >= opt.niter*0.9 and epoch % int(opt.save_epoch_freq/2) == 0) :
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(epoch)
    model.save('latest')