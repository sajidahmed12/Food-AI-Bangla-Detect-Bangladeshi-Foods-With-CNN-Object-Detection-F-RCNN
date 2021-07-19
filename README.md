# Food-AI-Bangla-Ver.-0.01-Detect-Bangladeshi-Foods-With-CNN-Object-Detection-F-RCNN

## Introduction

### This is my project for Pattern Recognition Course. 
### The Model Can detect total 4 Classes of Bangladeshi Foods. 


* chicken_curry
* khichuri
* rice
* tehari


# How to use this model 

## 1. Install Anaconda, CUDA, and cuDNN
* Download Anaconda for Windows from [their webpage](https://www.anaconda.com/products/individual)

* Download TensorFlow the [offical website](https://www.tensorflow.org/install)

* Download CUDA and CUdnn from [Here](https://www.tensorflow.org/install/source#tested_build_configurations)

Anaconda will automatically install the correct version of CUDA and cuDNN for the version of TensorFlow you are using.

---

## 2. Download TensorFlow Object Detection API repository from GitHub

| TensorFlow version | Models Repository Commit |
|--------------------|---------------------------|
|TF v1.12            |https://github.com/tensorflow/models/tree/r1.12.0 |
|TF v1.13            |https://github.com/tensorflow/models/tree/r1.13.0 |
|Latest version      |https://github.com/tensorflow/models |

This Project  done using TensorFlow v1.13 and this [GitHub commit](https://github.com/tensorflow/models/tree/079d67d9a0b3407e8d074a200780f3835413ef99) of the TensorFlow Object Detection API. If portions of this tutorial do not work, it may be necessary to install TensorFlow v1.13 and use this exact commit rather than the most up-to-date version.

---

## 3 . Download the Faster-RCNN-Inception-V2-COCO model from TensorFlow's model Zoo
Pre-trained classifiers with specific neural network architectures) in its [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md).


## 4 . Label Pictures 
it’s time to label the desired objects in every picture. LabelImg is a great tool for labeling images, and its GitHub page has very clear instructions on how to install and use it.

[LabelImg GitHub link](https://github.com/tzutalin/labelImg)

[LabelImg download link](https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1)

Download and install LabelImg, point it to your \images\train directory, and then draw a box around each object in each image. Repeat the process for all the images in the \images\test directory.

## 5. Generate Training Data
With the images labeled, it’s time to generate the TFRecords that serve as input data to the TensorFlow training model. Use the xml_to_csv.py and generate_tfrecord.py


## 6 Training

* move train.py from /object_detection/legacy into the /object_detection folder.

Here we go! From the \object_detection directory, run the following command to begin training:
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training faster_rcnn_inception_v2_pets.config
```

* Use tensorboard to view the steps and stages of training. 
```
(base) C:\dir\models\research\object_detection>tensorboard --logdir=training
```


## 7 Results 
 * Detecting Foods with our trained Object Detection Model. 
  
<p align="center">
  <img src="assets/result_gif.gif">
</p>

<p align="center">
  <img src="doc/detector2.jpg">
</p>


<p align="center">
  <img src="doc/detector2.jpg">
</p>

