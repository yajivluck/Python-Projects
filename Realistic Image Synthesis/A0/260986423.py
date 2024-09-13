# [TODO] Rename this file to YOUR-STUDENT-ID.py

##########################################
# DO NOT EDIT THESE IMPORT STATEMENTS!
##########################################
import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy, for now...
from pathlib import Path  # file path manipulation
##########################################


#####################
### Deliverable 1:
#####################
def render_checkerboard_slow(width, height, stride):
    image = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            if ((int(x / stride) + int(y / stride)) % 2) == 1:
                image[y, x] = 1
    return image


def render_checkerboard_fast(width, height, stride):
    ### BEGIN SOLUTION
    x,y = np.meshgrid(np.arange(width), np.arange(height))
    image = ((x // stride + y // stride) % 2).astype(np.uint8)
    
    return image # uncomment this line when you have implemented your solution
    ### END SOLUTION


#####################
### Deliverable 2
#####################
def circle(x, y):
    ### BEGIN SOLUTION
    value = np.power(x,2) + np.power(y,2) -np.power(0.5,2)
    return value # uncomment this line when you have implemented your solution
    ### END SOLUTION


def heart(x, y):
    ### BEGIN SOLUTION
    value =  np.power((np.power(x,2) + np.power(y,2) - 1),3) - np.power(x,2) * np.power(y,3) 
    return value # uncomment this line when you have implemented your solution
    ### END SOLUTION


def visibility_2d(scene_2d):
    ### BEGIN SOLUTION
    #Extract Bounds
    xmin,xmax,ymin,ymax = list(scene_2d['view'].values())
    
    width,height = scene_2d['output']['width'], scene_2d['output']['height']
    
    x_step = (xmax-xmin)/ width
    y_step = (ymax-ymin)/ height
    
    # Generate a grid of points within the specified bounds
    x = np.linspace(xmin, xmax, width, endpoint = False) + x_step/2 # Adjust the number of points as needed
    y = np.linspace(ymin, ymax, height, endpoint = False) + y_step/2 
    
    #x = np.arange(xmin, xmax, x_step)
    #y = np.arange(ymin, ymax, y_step)
    
    X, Y = np.meshgrid(x, y)
    
    image = scene_2d['shape_2d'](X,Y)
    #Convert implicit function evaluations to binary values of "inside" and "outside" shape
    image = image <= 0
    
    return image # uncomment this line when you have implemented your solution
    ### END SOLUTION

#####################
### Deliverable 3
#####################

def heart_3d(x, y):
    # Define the constants
    alpha = 9 / 4
    beta = 9 / 200

    x, y, z = get_heart_xyz(x, y, alpha, beta)

    values = get_heart_values(x, y, z, alpha, beta)

    normals = get_heart_normals(x, y, z, alpha, beta)

    return z, values, normals

def get_heart_xyz(x, y, alpha, beta):
    ### TO REPLACE FOR ECSE 546 STUDENTS ###
    file_path = Path(__file__).parent / 'heart_xyz.npy'
    x,y,z = np.load(str(file_path))
    return x,y,z
    ### TO REPLACE FOR ECSE 546 STUDENTS ###

def get_heart_values(x, y, z, alpha, beta):
    ### TO REPLACE FOR ECSE 546 STUDENTS ###
    file_path = Path(__file__).parent / 'heart_values.npy'
    values = np.load(str(file_path))
    return values
    ### TO REPLACE FOR ECSE 546 STUDENTS ###

def get_heart_normals(x, y, z, alpha, beta):
    ### TO REPLACE FOR ECSE 546 STUDENTS ###
    file_path = Path(__file__).parent / 'heart_normals.npy'
    normals = np.load(str(file_path))
    return normals
    ### TO REPLACE FOR ECSE 546 STUDENTS ###


def render(scene_3d):
    ### BEGIN SOLUTION
    xmin,xmax,ymin,ymax = list(scene_3d['view'].values())
    width,height = scene_3d['output']['width'], scene_3d['output']['height']
    x_step = (xmax-xmin)/ width
    y_step = (ymax-ymin)/ height
    # Generate a grid of points within the specified bounds
    x = np.linspace(xmin, xmax, width, endpoint = False) + x_step/2 # Adjust the number of points as needed
    y = np.linspace(ymin, ymax, height, endpoint = False) + y_step/2 
    X, Y = np.meshgrid(x, y)
    #Get z, values, and normals from heart 3d function
    z, default_image, normal = scene_3d['shape_3d'](X,Y)
    #Shaded Image where each pixel is 4 values (RGBa)
    image = np.zeros((width,height,4))
    # Extract light information into NumPy arrays
    light_colors = np.array([light['color'] for light in scene_3d['lights']])
    light_directions = np.array([light['direction'] for light in scene_3d['lights']])
    #Get Sum of all Light Directions as inner vector in width x height x 3 [R,G,B] 
    normal_dot_dir = np.matmul(normal, light_directions.T)
    #Bound R,G,B Values to 0 as a minimum
    np.maximum(normal_dot_dir, 0, out=normal_dot_dir)
    result_matrix = normal_dot_dir[:, :, :, np.newaxis] * light_colors
    image = np.sum(result_matrix,axis = 2)
    
    #Use z distance matrix as mask for alpha values (0 where inf, 1 otherwise)
    mask = ~np.isinf(z)
    image[..., 3] = mask.astype(float)
    np.clip(image,0,1)
      
    return image # uncomment this line when you have implemented your solution
    ### END SOLUTION





# Some example test routines for the deliverables. 
# Feel free to write and include your own tests here.
# Code in this main block will not count for credit, 
# but the collaboration and plagiarism policies still hold.
# You can change anything in the mainline -- it will not be graded
if __name__ == "__main__":

    # at some point, your code may (purposefully) propagate np.Inf values, 
    # and you may want to disable RuntimeWarnings for them; we will NOT penalize RuntimeWarnings,
    # _so long as your implementation produces the desired output_
    np.seterr(invalid='ignore')

    # convenience variable to enable/disable tests for different deliverables
    enabled_tests = [True, True, True]

    ##########################################
    ### Deliverable 1 TESTS
    ##########################################
    if enabled_tests[0]:
        # Test code to visualize the output of the functions
        plt.imshow(render_checkerboard_slow(256, 256, 3))
        plt.show()  # this is a *blocking* function call: code will not continue to execute until you close the plotting window!

        plt.imshow(render_checkerboard_fast(256, 256, 3))
        plt.show()

        plt.imshow(render_checkerboard_slow(256, 256, 3) - render_checkerboard_fast(256, 256, 3))
        plt.show()

        # import _anything you like_ but ONLY in the mainline for testing, not in your solution code above
        import time  # useful for performance measurement; see time.perf_counter()

        log_test_lengths = np.arange(3) + 2  # four orders of magnitude, starting from 100
        fast_perfs = []
        slow_perfs = []
        for length in log_test_lengths:
            start = time.perf_counter()
            #image_1 = render_checkerboard_fast(10 ** length, 10 ** length, 2)
            render_checkerboard_fast(10 ** length, 10 ** length, 2)

            end = time.perf_counter()
            fast_perfs.append(end - start)

            start = time.perf_counter()
            #image_2 = render_checkerboard_slow(10 ** length, 10 ** length, 2)
            render_checkerboard_slow(10 ** length, 10 ** length, 2)

            end = time.perf_counter()
            slow_perfs.append(end - start)
            
            #print((image_1 == image_2).all())

        plt.title("Checkerboard Performance Comparison")
        plt.xlabel("Output Image Side Length (in pixels)")
        plt.ylabel("Performance (in seconds)")
        plt.plot(10 ** log_test_lengths, slow_perfs, '-*')
        plt.plot(10 ** log_test_lengths, fast_perfs, '-x')
        plt.legend(['Naive', 'Fast'])
        plt.show()

    ##########################################
    ### Deliverable 2 TESTS
    ##########################################
    if enabled_tests[1]:
        test_scene_2d = {
            "output": {  # output image dimensions
                "width": 100,
                "height": 100
            },
            "shape_2d": circle,  # 2D shape function to query during visibility plotting
            "view": {  # 2D plotting limits
                "xmin": -1.,
                "xmax": 1.,
                "ymin": -1.,
                "ymax": 1.
            }
        }

        plt.imshow(visibility_2d(test_scene_2d))
        plt.show()

        test_scene_2d["shape_2d"] = heart
        test_scene_2d["view"]["xmin"] = -1.25
        test_scene_2d["view"]["xmax"] = 1.25
        test_scene_2d["view"]["ymin"] = 1.5
        test_scene_2d["view"]["ymax"] = -1.25
        plt.imshow(visibility_2d(test_scene_2d))
        plt.show()

    ##########################################
    ### Deliverable 3 TEST
    ##########################################
    if enabled_tests[2]:
        test_scene_3d = {
            "output": {  # output image dimensions
                "width": 100,
                "height": 100
            },
            "shape_3d": heart_3d,  # 3D shape function to query during rendering
            "lights":  # (directional) lights to use during shading
                [
                    {
                        "direction": np.array([-3, 3, -1]) / np.linalg.norm(np.array([-3, 3, -1])),
                        "color": np.array([1, 0.125, 0.125, 1])
                    },
                    {
                        "direction": np.array([3, 3, -1]) / np.linalg.norm(np.array([3, 3, -1])),
                        "color": np.array([0.125, 1.0, 0.125, 1])
                    },
                    {
                        "direction": np.array([0, -3, -1]) / np.linalg.norm(np.array([0, -3, -1])),
                        "color": np.array([0.125, 0.125, 1.0, 1])
                    }
                ],
            "view":  # 2D plotting limits (z limits are -infinity to infinity, i.e., consider all roots without clipping in z)
                {
                    "xmin": -1.25,
                    "xmax": 1.25,
                    "ymin": 1.5,
                    "ymax": -1.25
                }
        }
        
        
        #z,default_image,normal = render(test_scene_3d)
        image = render(test_scene_3d)

        plt.imshow(render(test_scene_3d))
        plt.show()