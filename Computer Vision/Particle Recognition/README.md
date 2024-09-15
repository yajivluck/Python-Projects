# Particle Contour Detection Script

This Python script processes images of microphones exposed to particles, detects contours of the particles, calculates their areas, and approximates their diameters. The script can save the contoured images and/or generate CSV files containing the diameters of detected particles. It uses OpenCV for image processing and contours detection.

## Features
- **Contour Detection**: Detects contours of particles in an image.
- **Diameter Approximation**: Approximates the diameter of each detected particle based on the area of the contour.
- **Save Results**: The script allows saving:
  - Contoured images with detected particles.
  - CSV files with particle diameters.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Pandas
- argparse (comes with Python by default)

Install dependencies using `pip`:
```bash
pip install opencv-python numpy pandas
```

## Usage
To run the script, use the following command:

```
python Particle_Vision.py [--save-images] [--save-csv]
```

At least one of the following arguments is required:

```
--save-images: Save the contoured images to the output folder.
--save-csv: Save the diameter CSV files to the output folder.
```

## Example Usage
```
python Particle_Vision.py --save-images
python Particle_Vision.py --save-csv
python Particle_Vision.py --save-images --save-csv
```