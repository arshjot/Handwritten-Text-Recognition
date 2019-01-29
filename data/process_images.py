import os
import glob
import subprocess
from tqdm import tqdm

if not os.path.exists('./IAM/processed_lines/'):
    os.makedirs('./IAM/processed_lines/')

for im_file in tqdm(glob.glob('./IAM/lines/*')):
    out_file = './IAM/processed_lines/'+im_file[im_file.rfind('/')+1:]
    command = ['imgtxtenh -d 118.110 -V 0 ' + im_file + ' png:- | \
        convert png:- -deskew 40% \
        -bordercolor white -border 5 -trim \
        -bordercolor white -border 20x0 +repage \
        -strip ' + out_file]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
