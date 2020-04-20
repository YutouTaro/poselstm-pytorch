##Windows
### compute image mean
python util/compute_image_mean.py --dataroot d:\Datasets\EuRoC\ --height 752 --width 480 --save_resized_imgs
### train
python train.py --model posenet --dataroot d:\Datasets\EuRoC\ --name d:\Datasets\EuRoC\beta500 --beta 500 --gpu 0 --init_weights ''
### test


##Workstation

###test
python3 test.py --model poselstm --dataroot /home/yu/DataSet/CambridgeLandmarks/KingsCollege/ --checkpoints_dir /home/yu/DataSet/CambridgeLandmarks/KingsCollege/ --name beta500 --gpu -1 --model posenet

### for ipython
import sys
sys.argv = "test.py --model poselstm --dataroot /home/yu/DataSet/CambridgeLandmarks/KingsCollege/ --checkpoints_dir /home/yu/DataSet/CambridgeLandmarks/KingsCollege/ --name beta500 --gpu -1 --model posenet".split()
