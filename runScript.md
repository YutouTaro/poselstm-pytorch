# for workstation
python3 test.py --model poselstm --dataroot /home/yu/DataSet/CambridgeLandmarks/KingsCollege/ --checkpoints_dir /home/yu/DataSet/CambridgeLandmarks/KingsCollege/ --name beta500 --gpu -1 --model posenet

# for ipython
import sys
sys.argv = "test.py --model poselstm --dataroot /home/yu/DataSet/CambridgeLandmarks/KingsCollege/ --checkpoints_dir /home/yu/DataSet/CambridgeLandmarks/KingsCollege/ --name beta500 --gpu -1 --model posenet".split()
