### this script converts the data of posenet format to euroc format for the evaluation tool "evo"

import argparse
import math
import numpy as np
import os
import sys

def params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filepath', required=True, help='the path of the file')
    # parser.add_argument('--height', type=int, default=256, help='image height')
    # parser.add_argument('--width', type=int, default=455, help='image width')
    # parser.add_argument('--save_resized_imgs', action="store_true", default=False, help='save resized train/test images [height, width]')
    return parser.parse_args()

args = params()

path_infile = args.filepath

if not os.path.exists(path_infile):
    print("%s does not exists." % (path_infile))
    sys.exit()

outfileName = os.path.dirname(path_infile).split('/')[-1] + '_' + os.path.basename(path_infile)
path_outfile = os.path.join("./", outfileName)


fin = open(path_infile, 'r')
fout= open(path_outfile,'w')
ctLine = 0
for line in fin:
    timestampStr, poseStr = line.split(' ', 1)
    timestampStr = timestampStr.split('/')[-1]
    timestampStr = timestampStr.split('.')[0]
    deci = len(timestampStr) - 10
    timestamp = int(timestampStr) / math.pow(10, deci)

    fout.write("%10.6f %s\n" % (timestamp, poseStr.strip()))
    ctLine += 1
fin.close()
fout.close()
print("Converting finished, %d lines converted\nData has been saved to %s" % (ctLine, path_outfile))