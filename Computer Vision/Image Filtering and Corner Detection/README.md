# Image Filtering and Corner Detection

## Introduction
This project is an introductory assignment as part of the **Introduction to Computer Vision (ECSE 415)** course. The project includes grayscaling, gaussian smoothing, gradient computations, edge magnitude and orientation as well as Canny Edge Detection. All these methods are applied to two images named Image_1.jpg and Image_2.jpg. To run this project on different images, replace the existing ones with two of your own. Make sure they are located in the same directory as the original, as well as with the same names as the original.

## Project Components

### 1. Image Acquisition and processing
Two images of a household object are taken from a different point of view and varying processes are applied to both images.

- Convert to Grayscale
- Smooth the Images using a 5x5 pixel Gaussian kernel as well as a 11x11 Gaussian kernel.
- Compute the x and y derivative images of the smoothed images using horizontal/vertical Sobel Filters.
- Compute the edge gradient magnitude/orientation of the smoothed images using the Sobel filter values
- Using opencv, compute the Canny Edge Detection of the image


## Prerequisites
Ensure that the following libraries are installed:
- **OpenCV**


## Running the Project
1. Clone this repository.
2. Open the Jupyter notebook file.
3. Define the `PATH` variable at the top of the notebook, pointing to the relative location of the two images to be used in this assignment. If you use the default images, or use your own images with the same name/location as the default images, you don't need to do anything here.
4. Run all cells to generate the required outputs.

## Notes
- Input and output images are displayed directly in the notebook.
- Ensure the code runs without errors before submission.
- Cite any external code sources used in the comments.
- Follow academic integrity guidelines as required by the course.

## Conclusion
This project demonstrates the practical application of a multitude of preprocessing techniques that can be applied to images. These different methods can help identify relevant information about images that could be used to lower the dimensions/computations required to properly process their data. This is useful in any application, especially model training.
