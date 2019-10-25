# Handwritten Text Recognition with TensorFlow
Code and model weights for English handwritten text recognition model trained on IAM Handwriting Database. 
It is more or less a TensorFlow port of [Joan Puigcerver's amazing work on HTR](https://github.com/jpuigcerver/Laia/tree/master/egs/iam).
This framework could also be used for building similar models using other datasets. 

## Requirements
- [ImageMagick](https://www.imagemagick.org/) - for image processing
- [imgtxtenh](https://github.com/mauvilsa/imgtxtenh) - for image processing
- [TensorFlow v1.12.0](https://www.tensorflow.org/) - for deep learning
- [OpenCV 3](https://pypi.org/project/opencv-python/) - for image processing (not required for prediction)
- (Optional) [TF implementation of STN (Spatial Transformer Network)](https://github.com/kevinzakka/spatial-transformer-network) - 
required for CRNN-STN architecture

## Steps for prediction on new images
1. Place the images of handwritten text in the `samples` folder
2. Download the model weights from **** and place it under the `experiments` directory.
Ensure that the directory structure is as below:
    ```bash
    ├── experiments
    │   ├── CRNN_h128
    │   │   ├── best_model
    │   │   ├── checkpoint
    │   │   └── summary
    ```
3. Enter the `mains` directory and run:
    ```bash 
    python predict.py -c ../configs/config.json
    ```

## Steps for training model from scratch
1. Download the [IAM dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) (you'll need to register on the website)
and keep the lines partition in the `/data/IAM/` directory as shown below:
    ```bash
    ├── data
    │   ├── IAM
    │   │   ├── lines
    │   │   │   ├── a01-000u-00.png
    │   │   │   ├── a01-000u-01.png
    │   │   │   ├── .
    │   │   │   ├── .
    │   │   │   ├── .
    │   │   ├── lines.txt
    ```
2. If required, modify the `/configs/config.json` file to change model architecture , image height, etc.
3. From the `data` directory, run:
    ```bash
    python process_images.py -c ../configs/config.json
    ```
    This will pre-process the images (add borders, resize, remove skew, etc.) 
    using imgtxtenh and ImageMagick's convert.
4. From the `data` directory, run:
    ```bash
    python prepare_IAM.py -c ../configs/config.json
    ```
    This will:
     - process the ground-truth labels to remove spaces within words and collapse contractions
     - read each image and create TFRecords files for train, validation and test sets using Aachen's partition

5. Start model training by running the below command from the `mains` directory:
    ```bash
    python main.py -c ../configs/config.json
    ```
## Results
XXXXXXXXXXXX

## Citations
- Laia: A deep learning toolkit for HTR
    ```
    @misc{laia2016,
      author = {Joan Puigcerver and
                Daniel Martin-Albo and
                Mauricio Villegas},
      title = {Laia: A deep learning toolkit for HTR},
      year = {2016},
      publisher = {GitHub},
      note = {GitHub repository},
      howpublished = {\url{https://github.com/jpuigcerver/Laia}},
    }
    ```
- Joan Puigcerver. [Are Multidimensional Recurrent Layers ReallyNecessary for Handwritten Text Recognition?](http://www.jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf) 
Pattern Recognition and Human Language Technology Research Center, Universitat Politècnica de València, Valencia, Spain
- U. Marti and H. Bunke. The IAM-database: An English Sentence Database for Off-line Handwriting Recognition. Int. Journal on Document Analysis and Recognition, Volume 5, pages 39 - 46, 2002.
- [TensorFlow implementation of Spatial Transformer Network](https://github.com/kevinzakka/spatial-transformer-network)
- [Mahmoud Gemy](https://github.com/MrGemy95) for providing the [Tensorflow Project Template](https://github.com/MrGemy95/Tensorflow-Project-Template).

## Status
*In Progress - model weights and results to be added*
