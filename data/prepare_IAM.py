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
import argparse
import tensorflow as tf
import pickle as pkl


def read_data(data_folder_path, out_height, out_name):
    """
    Return consolidated and preprocessed images, labels, image widths (without pad) and image label lengths
    split into training, validation and test sets
    Validation Set 2 will be added to Test Set

    Arguments:
    data_folder_path : Path of folder with lines.txt, 'lines' & 'largeWriterIndependentTextLineRecognitionTask' folders
    out_height       : Height of output image
    out_name        : Path and name of output tfrecords file
    """

    # Extract label text and IDs from lines.txt
    with open(data_folder_path + '/lines.txt', 'rb') as f:
        char = f.read().decode('unicode_escape')
        line_raw = char[1025:].splitlines()

    # Create dictionary of the format {line_id : [graylevel_for_binarizing, label_in_ascii]}
    # Lines with error in segmentation will be excluded
    chars = np.unique(np.concatenate([[char for char in line.split()[8]]
                                      for line in line_raw if line.split()[1] == 'ok']))
    char_map = {value: idx for (idx, value) in enumerate(chars)}
    char_map['<BLANK>'] = len(char_map)
    num_chars = len(char_map.keys())

    line_data = {line.split()[0]: [int(line.split()[2]), [char_map[char] for char in line.split()[8]]]
                 for line in line_raw if line.split()[1] == 'ok'}

    # Extract IDs for test, train and val sets
    with open(data_folder_path + '/aachen_partition/tr.lst', 'rb') as f:
        ids = f.read().decode('unicode_escape')
        train_ids = [i for i in ids.splitlines() if i in list(line_data.keys())]
    with open(data_folder_path + '/aachen_partition/va.lst', 'rb') as f:
        ids = f.read().decode('unicode_escape')
        val_ids = [i for i in ids.splitlines() if i in list(line_data.keys())]
    with open(data_folder_path + '/aachen_partition/te.lst', 'rb') as f:
        ids = f.read().decode('unicode_escape')
        test_ids = [i for i in ids.splitlines() if i in list(line_data.keys())]

    def convert_image(img_id, lines_data):
        img = cv2.imread(data_folder_path + '/processed_lines/' + img_id + '.png', 0)
        lab = np.array(lines_data[img_id][1], dtype=np.int32)

        # Threshold images to remove background and invert colors
        # TODO: Run below line if pre-processing = False and change path for above line
        # img[img > line_data[im_id][0]] = 255
        img = cv2.bitwise_not(img)
        img = np.divide(img.astype(np.float32), 255.0)

        # Resize - put a height cap and resize accordingly
        out_width = int((img.shape[1] / img.shape[0]) * out_height)
        img = cv2.resize(img, (out_width, out_height)).astype(np.float32)
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
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--data_dir", required=False,
                    help="Path of folder with lines.txt, lines & aachen_partition folders")
    ap.add_argument("-oh", "--out_height", required=False, type=int, help="Height of the output image")
    ap.add_argument("-o", "--output", required=False,
                    help="Output directory and name for tfrecords file (do not include extension)")
    args = vars(ap.parse_args())

    data_dir = './IAM/' if args['data_dir'] is None else args['data_dir']
    height = 64 if args['out_height'] is None else args['out_height']
    output = './iam' if args['output'] is None else args['output']

    print('Loading Data:')
    read_data(data_folder_path=data_dir, out_height=height, out_name=output + '_h' + str(height))
    print('Completed')
