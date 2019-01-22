"""
This file is for preparing IAM sentence level dataset
Please first download IAM sentence level dataset and extract it in a new folder here named 'IAM'
Ensure the following directory structure is followed:
├── data
|   ├── IAM
|       ├──lines.txt
|       └──largeWriterIndependentTextLineRecognitionTask
|           └──test, train, validation .txt files
|       └──lines
|           └──.png image files
|   └── prepare_IAM.py

Then run this script to prepare the data of IAM
"""
import sys

sys.path.extend(['..'])

import cv2
import numpy as np
from tqdm import tqdm
import argparse
import pickle


def read_data(data_folder_path, out_height, out_width):
    """
    Return consolidated and preprocessed images, labels, image widths (without pad) and image label lengths
    split into training, validation and test sets
    Validation Set 2 will be added to Test Set

    Arguments:
    data_folder_path : Path of folder with lines.txt, 'lines' & 'largeWriterIndependentTextLineRecognitionTask' folders
    out_height       : Height of output image
    out_width        : Width of output image
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
    with open(data_folder_path + '/largeWriterIndependentTextLineRecognitionTask/trainset.txt', 'rb') as f:
        ids = f.read().decode('unicode_escape')
        train_ids = [i for i in ids.splitlines() if i in list(line_data.keys())]
    with open(data_folder_path + '/largeWriterIndependentTextLineRecognitionTask/validationset1.txt', 'rb') as f:
        ids = f.read().decode('unicode_escape')
        val_ids = [i for i in ids.splitlines() if i in list(line_data.keys())]
    with open(data_folder_path + '/largeWriterIndependentTextLineRecognitionTask/testset.txt', 'rb') as f:
        ids = f.read().decode('unicode_escape')
        test_ids = [i for i in ids.splitlines() if i in list(line_data.keys())]
    with open(data_folder_path + '/largeWriterIndependentTextLineRecognitionTask/validationset2.txt', 'rb') as f:
        ids = f.read().decode('unicode_escape')
        test_ids += [i for i in ids.splitlines() if i in list(line_data.keys())]

    # For random order generation (shuffling)
    train_p = np.random.permutation(len(train_ids))
    val_p = np.random.permutation(len(val_ids))
    test_p = np.random.permutation(len(test_ids))

    # Import images and labels (map label IDs with label_dict)
    train_images, train_labels, train_im_widths, train_lab_lengths = np.zeros(
        shape=(len(train_ids), out_height, out_width), dtype=np.float32), [None] * len(train_ids), np.zeros(
        shape=(len(train_ids)), dtype=np.int32), np.zeros(shape=(len(train_ids)), dtype=np.int32)
    val_images, val_labels, val_im_widths, val_lab_lengths = np.zeros(
        shape=(len(val_ids), out_height, out_width), dtype=np.float32), [None] * len(val_ids), np.zeros(
        shape=(len(val_ids)), dtype=np.int32), np.zeros(shape=(len(val_ids)), dtype=np.int32)
    test_images, test_labels, test_im_widths, test_lab_lengths = np.zeros(
        shape=(len(test_ids), out_height, out_width), dtype=np.float32), [None] * len(test_ids), np.zeros(
        shape=(len(test_ids)), dtype=np.int32), np.zeros(shape=(len(test_ids)), dtype=np.int32)

    def read_img(img_id):
        img = cv2.imread(data_folder_path + '/lines/' + img_id + '.png', 0)

        # Threshold images to remove background and invert colors
        img[img > line_data[im_id][0]] = 255
        img = cv2.bitwise_not(img)
        img = np.divide(img.astype(np.float32), 255.0)

        # Resize - put a height cap and resize accordingly
        resize_width = int((img.shape[1] / img.shape[0]) * out_height)

        if resize_width < out_width:
            img_width = resize_width
            img = cv2.resize(img, (resize_width, out_height)).astype(np.float32)
            img = np.pad(img, ((0,0), (0, out_width-resize_width)), mode='constant')
        else:
            img_width = out_width
            img = cv2.resize(img, (out_width, out_height))

        return img, img_width

    train_im_num, val_im_num, test_im_num = 0, 0, 0
    for im_id in tqdm(train_ids+val_ids+test_ids):
        im, im_width = read_img(im_id)
        lab = np.array(line_data[im_id][1], dtype=np.int32)

        if im_id in train_ids:
            train_im_widths[train_p[train_im_num]] = im_width
            train_lab_lengths[train_p[train_im_num]] = len(lab)
            train_images[train_p[train_im_num]] = im
            train_labels[train_p[train_im_num]] = lab
            train_im_num += 1
        elif im_id in val_ids:
            val_im_widths[val_p[val_im_num]] = im_width
            val_lab_lengths[val_p[val_im_num]] = len(lab)
            val_images[val_p[val_im_num]] = im
            val_labels[val_p[val_im_num]] = lab
            val_im_num += 1
        elif im_id in test_ids:
            test_im_widths[test_p[test_im_num]] = im_width
            test_lab_lengths[test_p[test_im_num]] = len(lab)
            test_images[test_p[test_im_num]] = im
            test_labels[test_p[test_im_num]] = lab
            test_im_num += 1

    # train_labels_sparse, val_labels_sparse = sparse_tuple_from(train_labels), sparse_tuple_from(val_labels)
    # test_labels_sparse = sparse_tuple_from(test_labels)
    print(f'Number of characters = {num_chars}')
    print(f'Training Samples = {len(train_ids)}')
    print(f'Validation Samples = {len(val_ids)}')
    print(f'Test Samples = {len(test_ids)}')

    return {'train': {'images': train_images, 'labels': train_labels,
                      'im_widths': train_im_widths, 'lab_lengths': train_lab_lengths},
            'validation': {'images': val_images, 'labels': val_labels,
                           'im_widths': val_im_widths, 'lab_lengths': val_lab_lengths},
            'test': {'images': test_images, 'labels': test_labels,
                     'im_widths': test_im_widths, 'lab_lengths': test_lab_lengths},
            'num_chars' : num_chars, 'char_map': char_map}


if __name__ == '__main__':
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--data_dir", required=False,
                    help="Path of folder with lines.txt, lines & largeWriterIndependentTextLineRecognitionTask folders")
    ap.add_argument("-oh", "--out_height", required=False, type=int, help="Height of the output image")
    ap.add_argument("-ow", "--out_width", required=False, type=int, help="Width of the output image")
    ap.add_argument("-o", "--output", required=False,
                    help="Output directory and name for pickle file (do not include extension)")
    args = vars(ap.parse_args())

    data_dir = './IAM/' if args['data_dir'] is None else args['data_dir']
    height = 40 if args['out_height'] is None else args['out_height']
    width = 800 if args['out_width'] is None else args['out_width']
    output = './iam' if args['output'] is None else args['output']

    print('Loading Data:')
    data = read_data(data_folder_path=data_dir, out_height=height, out_width=width)

    print('Saving Data')
    out_name = output + '_h' + str(height) + '_w' + str(width) + '.pickle'
    with open(out_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print('Completed')
