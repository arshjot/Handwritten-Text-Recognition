import sys

sys.path.extend(['..'])

import os
import glob
import subprocess
from tqdm import tqdm
from utils.utils import get_args
from utils.config import process_config

# capture the config path from the run arguments, then process the json configuration file
try:
    args = get_args()
    config = process_config(args.config)
except:
    print("missing or invalid arguments")
    exit(0)

out_height = config.im_height
dataset = config.dataset
out_dir = './'+dataset+'/lines_h' + str(out_height) + '/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for im_file in tqdm(glob.glob('./'+dataset+'/lines/*')):
    out_file = out_dir + im_file[im_file.rfind('/')+1:]
    out_file = out_file[:out_file.rfind('.')] + '.jpg'
    command = ['imgtxtenh -d 118.110 -V 0 ' + im_file + ' png:- | \
        convert png:- -deskew 40% \
        -bordercolor white -border 5 -trim \
        -bordercolor white -border 20x0 \
        -resize x' + str(out_height) + ' +repage \
        -strip ' + out_file]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
