# Classifier, Object Recognition

## Introduction
This project is based on the concepts of classification in Computer Vision. At first, we use HoG (Histogram of Gradient) to classify basic NHL logos. Finally, a face Recognition System is implemented using three different classifiers: Linear Support Vector Machine (SVM), Random Forest, and k-Nearest Neighbors (KNN)

This project was developed as part of the **Introduction to Computer Vision (ECSE 415)** course.

## Project Components

### 1. Classification using HoG (25 points)
The Histogram of Gradient classifier is implemented as following:
- Resize the training images to 128 x 128
- Compute HoG features of size (32, 32, 8) for all training images.
- Apply blocknorm in 4 x 4 cell neighborhoods.
- Fit a nearest neighbor classifier with three neighbors (with KNeighborsClassifier from sklearn library)

The algorithm is applied to:
- 5 Montreal Canadians Logos 
- 5 Toronto Maple Leafs Logos

### 2. Face Recognition System (40 points)
In this section, publicly available Georgia Tech Face Database which comprises of images of 50 different individuals each represented by 15 color JPEG images are used. The following steps are taken to build the classifier:

1. Grayscale and Normalize each images to 128 x 192 pixels.
2. Use a 80/20 Training/Testing set split.
3. Implement the efficient Snapshot method for PCA (Principal Component Analysis) to create an eigenface representation for the images
4. Train and test the performance of multiple classifiers using the eigenface representation of each images in our dataset instead of the original images.

## Prerequisites
This project uses the following libraries:
- **OpenCV**
- **Scikit-Image**
- **NumPy**
- **sklearn**
- **pandas**

Make sure you have these installed in your environment before running the code.

## Running the Project
1. Clone this repository.
2. Open the Jupyter notebook file provided.
3. Ensure that the `path` variable at the beginning of the notebook points to the Data folder relative to the Jupyter NoteBook. By default, they are in the same directory so you don't need to do anything here.
4. Run all cells in the notebook to generate the results.

## Notes
- The input and output images are displayed directly in the Jupyter notebook. No external images need to be submitted.
- All code is original unless otherwise referenced in comments. Please make sure to cite any external code used in your submissions.
- The project was developed for the **ECSE 415** course assignment. Make sure to adhere to academic integrity guidelines when reusing or referencing the code.

## Conclusion
This project showcases various methods for preprocessing data as well as multiple different types of classifiers and their classifying performance in a Human Face Recognition System.
