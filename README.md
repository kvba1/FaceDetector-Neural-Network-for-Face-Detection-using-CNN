
# FaceDetector: Neural Network for Face Detection using CNN

## Overview

This project presents a neural network for face detection using Convolutional Neural Networks (CNN). The primary goal is to develop a model that not only identifies the presence of a face in an image but also predicts the coordinates of a bounding box around each detected face. The model utilizes a pre-trained VGG model as a feature extractor and custom-designed heads for classification and regression tasks.

## Table of Contents

1. [Theoretical Introduction](#1-theoretical-introduction)
2. [Data Preparation](#2-data-preparation)
    - [Capturing Images with a Webcam](#21-capturing-images-with-a-webcam)
    - [Applying Bounding Boxes with LabelMe](#22-applying-bounding-boxes-with-labelme)
    - [Data Augmentation](#23-data-augmentation)
3. [Model Architecture](#3-model-architecture)
4. [Model Training and Evaluation](#4-model-training-and-evaluation)
5. [Test Results](#5-test-results)
6. [Summary](#6-summary)

## 1. Theoretical Introduction

Face recognition is a critical task in the field of computer vision, with wide applications in security systems, monitoring, and human-computer interaction. The project aims to create a model that detects faces and predicts bounding box coordinates accurately. The model uses the pre-trained VGG network as a base feature extractor and custom heads for classification and regression.

Deep neural networks, especially CNNs, have been highly successful in computer vision tasks, including face recognition. Pre-trained models like VGG offer rich feature representations that can be fine-tuned on specific datasets, improving model accuracy and efficiency.

## 2. Data Preparation

### 2.1 Capturing Images with a Webcam

Images were captured using a webcam through a custom script, which allowed real-time image acquisition. The images were saved in the `train/images` and `test/images` directories in JPEG format.

### 2.2 Applying Bounding Boxes with LabelMe

LabelMe was used to manually annotate faces in the images by drawing bounding boxes. Each bounding box is defined by the coordinates of two points: the top-left corner (x_min, y_min) and the bottom-right corner (x_max, y_max). The annotations were saved as JSON files in the `train/labels` and `test/labels` directories.

### 2.3 Data Augmentation

To increase the diversity of the training data, various image augmentation techniques were applied using the Albumentations library. These techniques included random cropping, rotation, brightness and contrast adjustments, and RGB shifts.

## 3. Model Architecture

The FaceTracker model is designed for two main tasks: face classification and bounding box regression. It uses the pre-trained VGG model as a feature extractor, followed by custom-designed layers for classification and regression tasks.

The VGG model's convolutional layers process input images and generate rich feature representations, which are then used by the classification and regression heads to detect faces and predict bounding box coordinates.

## 4. Model Training and Evaluation

The model was trained using the Adam optimizer with cross-entropy loss for classification and SmoothL1Loss for bounding box regression. Training involved multiple epochs, during which the model learned both tasks from the processed images.

| Epoch | Class Loss | Regression Loss | Total Loss |
|-------|------------|-----------------|------------|
| 1     | 0.2119     | 0.013           | 0.2249     |
| 2     | 0.1082     | 0.007           | 0.1153     |
| ...   | ...        | ...             | ...        |
| 10    | 0.0099     | 0.0014          | 0.0113     |

## 5. Test Results

The model's performance was evaluated on a test dataset using visualizations and metrics. A function was written to detect faces in random batches from the test set and display the results.

The model's classification capability was evaluated using the ROC metric, showing near-perfect classification performance. The IoU metric was used to assess the model's bounding box regression performance, indicating that most bounding boxes were accurately placed.

## 6. Summary

The project successfully developed and evaluated a neural network model for face detection. The model was implemented using a deep neural network, trained on a well-prepared dataset, and tested on a separate test set. The dataset included images with annotated faces and corresponding bounding box coordinates. The model was trained using defined loss functions for classification and object localization. Evaluation was conducted on the test set using ROC and IoU metrics, demonstrating the model's effectiveness in face detection and bounding box localization.


## License

[MIT](https://choosealicense.com/licenses/mit/)

