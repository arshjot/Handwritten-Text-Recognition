"""
This file is for preparing IAM sentence level dataset
Please first download IAM sentence level dataset and extract it in a new folder here named 'IAM'
Ensure the following directory structure is followed:
├── data
|   ├── IAM
|       ├──lines.txt
|       └──lines
|           └──.png image files
|       └──aachen_partition
|           └──te.lst, tr.lst, va.lst
|   └── prepare_IAM.py

Then run this script to prepare the data of IAM
"""
import sys

sys.path.extend(['..'])

import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pickle as pkl
import re
from utils.utils import get_args
from utils.config import process_config


def read_data(data_folder_path, out_height, out_name):
    """
    Return consolidated and preprocessed images, labels, image widths (without pad) and image label lengths
    split into training, validation and test sets
    Aachen Partition will be used for splitting training, validation and test sets

    Arguments:
    data_folder_path : Path of folder with lines.txt, 'lines' & 'largeWriterIndependentTextLineRecognitionTask' folders
    out_height       : Height of output image
    out_name        : Path and name of output tfrecords file
    """

    # Extract label text and IDs from lines.txt
    with open(data_folder_path + '/lines.txt', 'rb') as f:
        char = f.read().decode('unicode_escape')
        line_raw = char[1025:].splitlines()

    # Create dictionary of the format {line_id : [graylevel_for_binarizing, label_as_numeric]}
    # 1. Remove spaces within words
    # 2. Replace "|" with whitespace
    # 3. Remove separations from contractions ("We 'll" -> "We'll")
    def collapse_contraction(match_pattern, string):
        return ''.join([j.lstrip() if match_pattern.match(j) else j.rstrip() 
                        for j in [i for i in match_pattern.split(string) if i]])

    patterns_list = [r'( \'t)', r'( \'m)', r'( \'ll)', r'( \'ve)', r'( \'s)', r'(\'re)', r'( \'d)', r'( \'T)',
                     r'( \'M)', r'( \'LL)', r'( \'VE)', r'( \'S)', r'(\'RE)', r'( \'D)']
    pattern = re.compile('|'.join(patterns_list))

    line_raw = [line.split() for line in line_raw]
    line_raw = [i[:8]+[collapse_contraction(pattern, ''.join(i[8:]).replace('|', ' '))] for i in line_raw]

    chars = np.unique(np.concatenate([[char for char in line[8]] for line in line_raw]))
    char_map = {value: idx for (idx, value) in enumerate(chars)}
    char_map['<BLANK>'] = len(char_map)
    num_chars = len(char_map.keys())

    line_data = {line[0]: [int(line[2]), [char_map[char] for char in line[8]]]
                 for line in line_raw}

    # Extract IDs for test, train and val sets
    with open(data_folder_path + '/aachen_partition/tr.lst', 'rb') as f:
        ids = f.read().decode('unicode_escape')
        train_ids = [i for i in ids.splitlines()]
    with open(data_folder_path + '/aachen_partition/va.lst', 'rb') as f:
        ids = f.read().decode('unicode_escape')
        val_ids = [i for i in ids.splitlines()]
    with open(data_folder_path + '/aachen_partition/te.lst', 'rb') as f:
        ids = f.read().decode('unicode_escape')
        test_ids = [i for i in ids.splitlines()]

    def convert_image(img_id, lines_data):
        img = cv2.imread(data_folder_path + '/lines_h' + str(out_height) + '/' + img_id + '.jpg', 0)
        lab = np.array(lines_data[img_id][1], dtype=np.int32)

        # Threshold images to remove background and invert colors
        img = cv2.bitwise_not(img)
        img = np.divide(img.astype(np.float32), 255.0)
        out_width = img.shape[1]
        img = cv2.imencode('.png', img)[1].tostring()

        example_img = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=lab)),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[out_width])),
            'lab_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(lab)]))
        }))

        return example_img

    print('Training Data')
    with tf.python_io.TFRecordWriter(out_name+'_train.tfrecords') as writer:
        for im_id in tqdm(train_ids):
            example = convert_image(im_id, line_data)
            writer.write(example.SerializeToString())
    print('Validation Data')
    with tf.python_io.TFRecordWriter(out_name+'_val.tfrecords') as writer:
        for im_id in tqdm(val_ids):
            example = convert_image(im_id, line_data)
            writer.write(example.SerializeToString())
    print('Test Data')
    with tf.python_io.TFRecordWriter(out_name+'_test.tfrecords') as writer:
        for im_id in tqdm(test_ids):
            example = convert_image(im_id, line_data)
            writer.write(example.SerializeToString())

    with open(out_name+'_char_map.pkl', 'wb') as f:
        pkl.dump({'char_map': char_map, 'num_chars': num_chars, 'len_train': len(train_ids),
                 'len_val': len(val_ids), 'len_test': len(test_ids)}, f, protocol=pkl.HIGHEST_PROTOCOL)

    print(f'Number of characters = {num_chars}')
    print(f'Training Samples = {len(train_ids)}')
    print(f'Validation Samples = {len(val_ids)}')
    print(f'Test Samples = {len(test_ids)}')


if __name__ == '__main__':
    # capture the config path from the run arguments, then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    data_dir = './'+config.dataset+'/'
    height = config.im_height
    output = './'+config.dataset

    print('Loading Data:')
    read_data(data_folder_path=data_dir, out_height=height, out_name=output + '_h' + str(height))
    print('Completed')
