# Feature Extraction Project

## Introduction
This project is based on the concepts of feature extraction in computer vision, focusing on key methods such as Harris Corner Detection, SIFT (Scale-Invariant Feature Transform), and Image Stitching. The main goal is to implement these methods, analyze their outputs, and understand their behavior under different conditions.

This project was developed as part of the **Introduction to Computer Vision (ECSE 415)** course.

## Project Components

### 1. Harris Corner Detection (20 points)
The Harris Corner Detection method is implemented to detect corners in different images. The steps include:
- Computing image derivatives.
- Applying Gaussian filtering.
- Calculating the cornerness function response.
- Performing non-maximum suppression.

The algorithm is applied to:
- A checkerboard image, with varying threshold values.
- A building image, to analyze different threshold effects.

### 2. SIFT Features (40 points)
SIFT is a method to detect and describe local features in images. This part explores its invariance under scale and rotation. The main tasks are:
1. **SIFT in a Nutshell**: A brief description of the four main actions in the SIFT algorithm.
2. **SIFT between Two Different Pictures**: Matching keypoints between two images using brute-force methods.
3. **Invariance Under Scale**: Verifying the invariance of SIFT features with scaled versions of an image.
4. **Invariance Under Rotation**: Analyzing the SIFT features under different rotation angles.

### 3. Image Stitching (30 points)
The goal is to stitch three different views of the same scene together:
- Compute SIFT keypoints for matching.
- Find homographies using the RANSAC method.
- Apply transformations and perform linear image blending to create a panorama.
- A discussion on when to use pyramid blending over linear blending is included.

## Prerequisites
This project uses the following libraries:
- **OpenCV**
- **Scikit-Image**
- **NumPy**

Make sure you have these installed in your environment before running the code.

## Running the Project
1. Clone this repository.
2. Open the Jupyter notebook file provided.
3. Ensure that the `path` variable at the beginning of the notebook points to your working directory. For example:

    ```python
    path = '/path/to/images/'
    ```

4. Run all cells in the notebook to generate the results.

## Notes
- The input and output images are displayed directly in the Jupyter notebook. No external images need to be submitted.
- All code is original unless otherwise referenced in comments. Please make sure to cite any external code used in your submissions.
- The project was developed for the **ECSE 415** course assignment. Make sure to adhere to academic integrity guidelines when reusing or referencing the code.

## Conclusion
This project showcases various feature extraction methods and provides insights into their applications and performance under different transformations. By completing the project, a better understanding of corner detection, SIFT features, and image stitching was achieved.
