# [TODO] Rename this file to YOUR-STUDENT-ID.py

##########################################
# DO NOT EDIT THESE IMPORT STATEMENTS!
##########################################
import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy
##########################################

def normalize(v):
    """
    Returns the normalized vector given vector v.
    Note - This function is only for normalizing 1D vectors instead of batched 2D vectors.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


### BEGIN SOLUTION
def look_at_inv(eye, at, up):
    """
    Returns the inverse of the look_at matrix, given the eye location,
    the eye gaze direction and the up direction.
    The inverse of the look_at matrix transforms 3D homogeneous coordinates from
    the camera/view frame to the world frame.
    """
    
    z_c = normalize(at-eye)
    #normalize up in case it is not
    up = normalize(up)
    x_c = np.cross(up,z_c)
    y_c = np.cross(z_c,x_c)
    
    #Build transformation matrix from camera space to world space
    matrix = np.column_stack((x_c, y_c, z_c, eye))
    # Add a row of zeros at the bottom
    zeros_row = np.zeros((1, matrix.shape[1]))
    matrix = np.vstack((matrix, zeros_row))
    #Make 1 at the bottom right of the matrix as in shown in the assignment instructions
    matrix[3][3] = 1
    
    #Return the matrix function
    return matrix
     
 
    
### END SOLUTION


# ray bundles
class Rays(object):

    def __init__(self, Os, Ds):
        """
        Initializes a bundle of rays containing the rays'
        origins and directions.
        """
        self.Os = np.copy(Os)
        self.Ds = np.copy(Ds)

    def __call__(self, t):
        """
        Computes an array of 3D locations given the distances
        to the points.
        """
        return self.Os + self.Ds * t[:, np.newaxis]

    def __str__(self):
        return "Os: " + str(self.Os) + "\n" + "Ds: " + str(self.Ds) + "\n"

    def distance(self, point):
        """
        Compute the distances from the ray origins to a point
        """
        return np.linalg.norm(point[np.newaxis, :] - self.Os, axis=1)

# abstraction for every scene object
class Geometry(object):
    def __init__(self):
        return

    def intersect(self, rays):
        return
    
# sphere objects for our scene
class Sphere(Geometry):
    EPSILON_SPHERE = 1e-4

    def __init__(self, r, c, brdf_params):
        """
        Initializes a sphere object with its radius, position and albedo.
        """
        self.r = np.float64(r)
        self.c = np.copy(c)
        self.brdf_params = brdf_params
        super().__init__()

    def intersect(self, rays):
        """
        Intersect the sphere with a bundle of rays, and compute the
        distance between the hit point on the sphere surface and the
        ray origins. If a ray did not intersect the sphere, set the
        distance to np.inf.
        """
        ### BEGIN SOLUTION
      
        # Define coefficient for quadratic to solve
        A = 1
        B = 2 * np.sum((rays.Os - self.c) * rays.Ds, axis=1)
        C = np.sum((rays.Os - self.c)**2, axis=1) - self.r**2
        
        # Calculate the discriminants
        discriminants = B**2 - 4*A*C
        
        # Initialize the template time vector with default no contact (np.inf)
        t = np.full(discriminants.shape, np.inf)
        
        # Mask for zero discriminants based on epsilon
        zero = discriminants == 0
        
        # For discriminants close to zero, the solution is easier to compute
        t[zero] = -B[zero] / (2 * A)
        
        # Mask for positive discriminants
        positive_discriminant = discriminants > 0
        
        # Calculate t1 and t2
        t1 = (-B[positive_discriminant] - np.sqrt(discriminants[positive_discriminant])) / (2 * A)
        t2 = (-B[positive_discriminant] + np.sqrt(discriminants[positive_discriminant])) / (2 * A)
        
        # Create masks for non-negative t1 and t2
        non_negative_t1 = t1 >= 0
        non_negative_t2 = t2 >= 0
        
        # Use np.where to classify the combination of t solns (based on whether they are positive or negative)
        t[positive_discriminant] = np.where(
            non_negative_t1 & non_negative_t2, np.minimum(t1, t2),      # both t1 and t2 non-negative
            np.where(non_negative_t1, t1,                               # only t1 is non-negative
                     np.where(non_negative_t2, t2, np.inf))             # only t2 is non-negative
        )
        
        
        # Return only the non-negative values of t (greater than EPISLON_SPHERE )
        t[t<self.EPSILON_SPHERE] = np.inf
        return t 


        ### END SOLUTION


class Scene(object):

    def __init__(self, w, h):
        """ Initialize the scene. """
        self.w = w
        self.h = h

        # Camera parameters. Set using set_camera_parameters() 
        self.eye = np.empty((3,), dtype=np.float64)
        self.at = np.empty((3,), dtype=np.float64)
        self.up = np.empty((3,), dtype=np.float64)
        self.fov = np.inf

        # Scene objects. Set using add_geometries()
        self.geometries = []

        # Light sources. Set using add_lights()
        self.lights = []

    def set_camera_parameters(self, eye, at, up, fov):
        """ Sets the camera parameters in the scene. """
        self.eye = np.copy(eye)
        self.at = np.copy(at)
        self.up = np.copy(up)
        self.fov = np.float64(fov)

    def add_geometries(self, geometries):
        """ Adds a list of geometries to the scene. """
        self.geometries.extend(geometries)

    def add_lights(self, lights):
        """ Adds a list of lights to the scene. """
        self.lights.extend(lights)

    def generate_eye_rays(self):
        """
        Generate a bundle of eye rays.

        The eye rays originate from the eye location, and shoots through each
        pixel into the scene.
        """
        
        ### BEGIN SOLUTION
        
        # Making x/y coords of plane (normalized from -1 to 1) 
        # and centered to stratum
        w,h = self.w,self.h
        
        x_step,y_step = 2/w, 2/h
        
        
        x_coords = np.linspace(-1,1,w,endpoint = False) + (x_step / 2)
        #Invert y coords to make meshgrid match with out coords system in NDC
        y_coords = np.linspace(1, -1, h, endpoint = False) - (y_step / 2)
        
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
        
        #Coordinate axis in camera space
        z_c = normalize(self.at - self.eye)
        x_c = normalize(np.cross(normalize(self.up),z_c))
        y_c = np.cross(z_c,x_c)
        
        #height fov convert degree to randians and take tan to get the scale from NDC->Camera
        height_fov = np.tan(np.radians(self.fov/2))  
        #Width fov as aspect ratio times height fov
        width_fov = height_fov * w / h
        
        camera_x = x_mesh[:,:,np.newaxis] * x_c * width_fov
        camera_y = y_mesh[:,:,np.newaxis] * y_c * height_fov    
        camera_space_coords = camera_x + camera_y + z_c
        
        #print(camera_space_coords.shape)
    
        # Create an array of zeros with shape (768, 1024, 1)        
        zeros = np.zeros(camera_space_coords.shape[:-1] + (1,))
        #Add a 4th element to inner vectors of camera coords to fit lookatinv transform
        camera_space_coords = np.concatenate((camera_space_coords, zeros), axis=2)
        
        #Normalize camera space vectors
        camera_space_coords_norms = np.linalg.norm(camera_space_coords, axis=-1, keepdims=True)
        camera_space_coords = camera_space_coords / camera_space_coords_norms
        
    
        #Convert to world space using look_at_inv
        world_space_coords = np.dot(camera_space_coords,look_at_inv(eye = self.eye, at = self.at, up = self.up))
        #Remove last element (4 element not needed)
        world_space_coords = world_space_coords[:, :, :-1]
        #Reshape world coords into list of vector coords
        world_space_coords = np.reshape(world_space_coords, (-1,3))
                
        Bundle_of_rays = Rays(Os = np.asarray([self.eye]), Ds = world_space_coords)
        return Bundle_of_rays
        
    
        

        ### END SOLUTION

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """
        hit_ids = np.array([-1])
        hit_distances = np.array([np.inf])
        hit_normals = np.array([np.inf, np.inf, np.inf])

        ### BEGIN SOLUTION
        
        hit_distances = []
        
        Os_shape,Ds_shape = rays.Os.shape,rays.Ds.shape
        
        normal_shape = max(Os_shape,Ds_shape)
        
        hit_normals = np.full(normal_shape, np.inf)
        
        #Build centers list that will hold centers based on number of sphere in scene
        centers = np.empty((0,len(self.geometries)))
        
        radii = []
        
        #Enumerate for loop for hit_it and sphere
        for hit_id, sphere in enumerate(self.geometries):
            #Get the times (as distance) that the sphere was hit by the rays
            hit_distance = sphere.intersect(rays)
                        
            hit_distances.append(hit_distance)
            #Append centers of sphere to centers list
            centers = np.vstack([centers, sphere.c])
            radii.append(sphere.r)

        #convert to numpy array  
        radii = np.asarray(radii)
        
        # Convert the list of arrays into a 2D numpy array, zipped together
        zipped_distances = np.column_stack(hit_distances)
        
        
        # Get the indices of the smallest values in each inner list
        #This is the true hit distance per array
        hit_ids = np.argmin(zipped_distances, axis=1)
        
        
        # Get the smallest values in each inner list. This is the true hit_distances
        hit_distances = zipped_distances[np.arange(zipped_distances.shape[0]), hit_ids]
        #hit_distances -> 786432,
        
        #Replace hit ids with -1 when distance is not a hit        
        hit_ids[hit_distances == np.inf] = -1  
        #id mask
        id_mask = hit_ids != -1
        
        #centers[hit_ids] -> 786432,3
        hit_normals[id_mask] = (centers[hit_ids][id_mask] - rays(hit_distances)[id_mask]) / radii[hit_ids][id_mask][:, np.newaxis]
            
    
        ### END SOLUTION
        return hit_distances, hit_normals, hit_ids
    
    # Shade a scene given a bundle of eye rays; outputs a color image suitable for matplotlib visualization
    def shade(self, rays):
        shadow_ray_o_offset = 1e-6
        ### BEGIN SOLUTION
             
        #Set intensity to accept RGB (default at zero)
        L = np.zeros((self.h*self.w, 3))
        #intersect with scene
        hit_distances, hit_normals, hit_ids = self.intersect(rays)
        id_mask = hit_ids != -1
                    
        #Make numpy arr of geometries
        geometries = np.asarray(self.geometries)
        ro = np.array([geometry.brdf_params[:3] for geometry in geometries])
        
        for light in self.lights:
            
            #Direction/Color of single light 
            direction = normalize(light['direction'])
            color = light['color']
            
            #Shadow ray of specific light source direction is opposite to light source offsetted
            #Source of shadow is from camera called at hit distance offsetted by the normal
            shadow_rays = Rays(Os = rays(hit_distances) + hit_normals * shadow_ray_o_offset, Ds = [direction])
                        
            #Intercept with shadow rays
            shadow_hit, shadow_normals, shadow_ids = self.intersect(shadow_rays)
            
            #Mask for shadow ray that don't hit objects
            shadow_mask = shadow_ids != -1
            
            #Only add light intensity if light hits (id_mask) AND no shadow (shadow_mask)
            L[id_mask & ~shadow_mask] += ro[hit_ids][id_mask & ~shadow_mask] * (np.maximum(0,np.dot(hit_normals[id_mask & ~shadow_mask],-direction))[:,np.newaxis] * color[:3] / np.pi)
            
            

           
        # Correcting potential clipping errors
        L = np.clip(L, 0, 1)
        ### END SOLUTION
        L = L.reshape((self.h, self.w, 3))
        return L


if __name__ == "__main__":
        
    enabled_tests = [True,True,True,True,False]

    ##########################################
    ### Deliverable 1 TESTS Rays Sphere Intersection
    ##########################################
    if enabled_tests[0]:
        # Point tests for ray-sphere intersection
        sphere = Sphere(1, np.array([0, 0, 0]), brdf_params=np.array([0.0, 0.0, 0.0, 1.0]))
        rays = Rays(np.array([
            # Moving ray origin along y-axis with x, z axis fixed
            [0, 2, -2],  # should not intersect
            [0, 1, -2],  # should intersect once (tangent)
            [0, 0, -2],  # should intersect twice
            [0, -1, -2],  # should intersect once (bottom)
            [0, -2, -2],  # should not intersect
            # Move back along the z-axis
            [0, 0, -4],  # should have t 2 greater than that of origin [0, 0, -2]
        ]), np.array([[0, 0, 1]]))

        expected_ts = np.array([np.inf, 2, 1, 2, np.inf, 3], dtype=np.float64)
        hit_distances = sphere.intersect(rays)

        if np.allclose(hit_distances, expected_ts):
            print("Rays-Sphere Intersection point test passed")
        else:
            raise ValueError(f'Expected intersection distances {expected_ts}\n'
                             f'Actual intersection distances {hit_distances}')

    ##########################################
    ### Deliverable 2 TESTS Eye Ray Generation
    ##########################################
    if enabled_tests[1]:
        # Create test scene and test sphere
        scene = Scene(w=1024, h=768)
        scene.set_camera_parameters(
            eye=np.array([0, 0, -10], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        sphere = Sphere(10, np.array([0, 0, 50]), brdf_params=np.array([0.0, 0.0, 0.0, 1.0]))

        
        vectorized_eye_rays = scene.generate_eye_rays()
        hit_distances = sphere.intersect(vectorized_eye_rays)

        # Visualize hit distances
        plt.matshow(hit_distances.reshape((768, 1024)))
        plt.title("Distances")
        plt.colorbar()
        plt.show()

    ##########################################
    ### Deliverable 3 TESTS Rays Scene Intersection
    ##########################################
    if enabled_tests[2]:
        # Set up scene
        scene = Scene(w=1024, h=768)
        scene.set_camera_parameters(
            eye=np.array([0, 0, -10], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        scene.add_geometries([
            # x+ => right; y+ => up; z+ => close to camera
            # Left Sphere in the image
            Sphere(16.5, np.array([-30, -22.5, 140]), brdf_params=np.array([0.999, 0.5, 0.5, 1.0])),
            # Right Sphere in the image
            Sphere(16.5, np.array([22, -27.5, 140]), brdf_params=np.array([0.5, 0.999, 0.5, 1.0])),
            # Ground
            # Sphere(1650, np.array([23, -1680, 170]), f=np.array([0.7, 0.7, 0.7])),
            Sphere(1650, np.array([23, -1700, 140]), brdf_params=np.array([0.7, 0.7, 0.7, 1.0])),
        ])

        vectorized_eye_rays = scene.generate_eye_rays()
        hit_distances, hit_normals, hit_ids = scene.intersect(vectorized_eye_rays)

        # Visualize distances, normals and IDs
        plt.matshow(hit_distances.reshape((768, 1024)))
        plt.title("Distances")
        plt.show()
        plt.matshow(np.abs(hit_normals.reshape((768, 1024, 3))))
        plt.title("Normals")
        plt.show()
        plt.matshow(hit_ids.reshape((768, 1024)))
        plt.title("IDs")
        plt.show()

    ##########################################
    ### Deliverable 4 TESTS Shading with Directional Light
    ##########################################
    if enabled_tests[3]:
        # Set up scene
        scene = Scene(w=1024, h=768)
        scene.set_camera_parameters(
            eye=np.array([0, 0, -10], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        scene.add_geometries([
            # x+ => right; y+ => up; z+ => close to camera
            # Left Sphere in the image
            Sphere(16.5, np.array([-30, -22.5, 140]), brdf_params=np.array([0.999, 0.5, 0.5, 1.0])),
            # Right Sphere in the image
            Sphere(16.5, np.array([22, -27.5, 140]), brdf_params=np.array([0.5, 0.999, 0.5, 1.0])),
            # Ground
            Sphere(1650, np.array([23, -1700, 140]), brdf_params=np.array([0.7, 0.7, 0.7, 1.0])),
        ])
        scene.add_lights([
            {
                "type": "directional",
                # Top-Left of the scene
                "direction": normalize(np.array([1, 1, 0])),
                "color": np.array([2, 0, 0, 1])  # Red
            },
            {
                "type": "directional",
                # Top-Right of the scene
                "direction": normalize(np.array([-1, 1, 0])),
                "color": np.array([0, 2, 0, 1])  # Green
            },
            {
                "type": "directional",
                # Top of the scene
                "direction": normalize(np.array([0, 1, 0])),
                "color": np.array([2, 2, 2, 1])  # White
            },
        ])

        vectorized_eye_rays = scene.generate_eye_rays()
        L = scene.shade(vectorized_eye_rays)

        plt.matshow(L)
        plt.title("Rendered Image")
        # plt.savefig("numpy-image.png")
        plt.show()

    ##########################################
    ### Deliverable 5 TESTS Shading with Point Light
    ##########################################
    if enabled_tests[4]:
        # Set up scene
        scene = Scene(w=1024, h=768)
        scene.set_camera_parameters(
            eye=np.array([0, 0, -10], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        scene.add_geometries([
            # x+ => right; y+ => up; z+ => close to camera
            # Left Sphere in the image
            Sphere(16.5, np.array([-30, -22.5, 140]), brdf_params=np.array([0.999, 0.5, 0.5, 1.0])),
            # Right Sphere in the image
            Sphere(16.5, np.array([22, -27.5, 140]), brdf_params=np.array([0.5, 0.999, 0.5, 1.0])),
            # Ground
            Sphere(1650, np.array([23, -1700, 140]), brdf_params=np.array([0.7, 0.7, 0.7, 1.0])),
        ])
        scene.add_lights([
            {
                "type": "point",
                # Top Left
                "position": np.array([-50, 30, 140], dtype=np.float64),
                "color": np.array([1e5, 0, 0, 1])  # Red
            },
            {
                "type": "point",
                # Top Right
                "position": np.array([50, 30, 140], dtype=np.float64),
                "color": np.array([0, 1e5, 0, 1])  # Green
            },
            {
                "type": "point",
                # Between the spheres
                "position": np.array([-4, -25, 140], dtype=np.float64),
                "color": np.array([1e4, 1e4, 1e4, 1])  # White
            },
            {
                "type": "point",
                # Center but closer to brighten up the scene
                "position": np.array([0, 30, 90], dtype=np.float64),
                "color": np.array([1e5, 1e5, 1e5, 1])  # White
            },
        ])

        vectorized_eye_rays = scene.generate_eye_rays()
        L = scene.shade(vectorized_eye_rays)

        plt.matshow(L)
        plt.title("Rendered Image")
        # plt.savefig("numpy-image.png")
        plt.show()
