# Feature Extraction Project

## Introduction
This project is a feature extraction assignment as part of the **Introduction to Computer Vision (ECSE 415)** course. The project includes tasks related to Harris Corner Detection, SIFT (Scale-Invariant Feature Transform), and Image Stitching to extract and match features from images, with a focus on their application in real-world computer vision tasks.

## Project Components

### 1. Harris Corner Detection (20 points)
The Harris Corner Detection algorithm is implemented to identify corners in images through the following steps:
- Computing image derivatives (optionally applying a blur).
- Squaring the derivatives.
- Applying Gaussian filtering.
- Calculating the cornerness function response using the determinant and trace of the image gradient matrix.
- Performing non-maximum suppression to localize the detected corners.

The algorithm is tested on:
- A checkerboard image, with variable thresholds.
- A building image, with threshold experimentation and observations.

### 2. SIFT Features (40 points)
This part explores the Scale-Invariant Feature Transform (SIFT), including matching features across different images and testing the method’s robustness to scale and rotation transformations.

#### Tasks:
- **SIFT in a Nutshell**: A brief explanation of the four main actions of the SIFT algorithm.
- **SIFT between Two Different Pictures**: Computing and matching SIFT keypoints between two images using a brute-force method and visualizing the top matches.
- **Invariance Under Scale**: Testing the SIFT feature invariance with scaled versions of the same image.
- **Invariance Under Rotation**: Evaluating the effect of rotation on SIFT feature matching.

### 3. Image Stitching (30 points)
In this section, three images of a scene are stitched together to create a panorama:
1. Compute SIFT keypoints for image pairs.
2. Find the best keypoint matches and compute the homography using RANSAC.
3. Transform the images and stitch them together using linear image blending.
4. A discussion on the possible use of pyramid blending is included.

## Prerequisites
Ensure that the following libraries are installed:
- **OpenCV**
- **Scikit-Image**
- **NumPy**

## Running the Project
1. Clone this repository.
2. Open the Jupyter notebook file.
3. Define the `PATH` variable at the top of the notebook, pointing to the relative location of the `Data` folder containing the images used in the assignment. By default, this value is empty since the Data Folder is found in the same directory as the Jupyter NoteBook.
4. Run all cells to generate the required outputs.

## Notes
- Input and output images are displayed directly in the notebook.
- Ensure the code runs without errors before submission.
- Cite any external code sources used in the comments.
- Follow academic integrity guidelines as required by the course.

## Conclusion
This project demonstrates the practical application of several feature extraction methods used in computer vision, highlighting their robustness to different transformations and their role in creating image panoramas.
