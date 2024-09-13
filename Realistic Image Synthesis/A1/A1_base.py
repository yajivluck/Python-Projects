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

        ### END SOLUTION
        return hit_distances, hit_normals, hit_ids
    
    # Shade a scene given a bundle of eye rays; outputs a color image suitable for matplotlib visualization
    def shade(self, rays):
        shadow_ray_o_offset = 1e-6
        ### BEGIN SOLUTION

        ### END SOLUTION
        L = L.reshape((self.h, self.w, 3))
        return L


if __name__ == "__main__":
    enabled_tests = [True, True, True, True, False]

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
