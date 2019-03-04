import os
import glob
import subprocess
from tqdm import tqdm

out_height = 128
out_dir = './IAM/lines_h' + str(out_height) + '/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for im_file in tqdm(glob.glob('./IAM/lines/*')):
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
