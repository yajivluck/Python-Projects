# Image Segmentation and Object Detection

This project involves various techniques for image segmentation and object detection using Python. It is divided into two main parts:

## Part 1: K-Means and Mean-Shift Clustering for Segmentation

In this part, we perform image segmentation using clustering techniques. The process includes:

1. **Convolution with Haar Filters:**
   - Convolve two images (`Person.jpg` and `Landscape.png`) with Haar filter kernels with zero padding.

2. **K-Means Clustering:**
   - Implement K-Means clustering to segment the images with \( K = 3 \).

3. **Mean-Shift Clustering:**
   - Implement Mean-Shift clustering to segment both images using the `scikit-learn` library.

## Part 2: Neural Network Implementation for Image Segmentation

In this part, we use a pre-trained neural network model for object detection. The process includes:

1. **Mask R-CNN:**
   - Utilize a pre-trained Mask R-CNN model to output bounding boxes of detected objects along with their object categories.

2. **Application:**
   - Run the Mask R-CNN model on two images: `street.png` and `Montreal_Picture.jpg` (a picture taken in Montreal).

## Setup Instructions

1. **Dependencies:**
   - Ensure that all required libraries are installed. Most dependencies are handled via pip, and the specific versions of each library are managed to ensure compatibility.

2. **Image Files:**
   - Place the following images in the directory specified by the `PATH` variable:
     - `Person.png`
     - `Landscape.png`
     - `street.png`
     - `Montreal_Picture.jpg` (Make sure this image is named exactly as specified)

3. **Running the Code:**
   - Modify the `PATH` variable in the notebook to point to the directory where the image files are located.
   - Execute the notebook to perform the image segmentation and object detection tasks as described.

## Notes

- Make sure to use the exact filenames for the images to avoid any issues.
- All necessary imports and their versions are handled automatically for smooth execution.

Feel free to reach out if you encounter any issues or have any questions.

