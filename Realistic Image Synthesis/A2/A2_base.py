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

def look_at_inv(eye, at, up):
    """
    Returns the inverse of the look_at matrix, given the eye location,
    the eye gaze direction and the up direction.
    The inverse of the look_at matrix transforms 3D homogeneous coordinates from
    the camera/view frame to the world frame.
    """
    # Copy solution from A1
    ### BEGIN SOLUTION
    assert False, "Remove assert and complete this section."
    ### END SOLUTION

# ray bundles
class Rays(object):

    def __init__(self, Os, Ds):
        """
        Initializes a bundle of rays containing the rays'
        origins and directions. Explicitly handle broadcasting
        for ray origins and directions; they must have the same 
        size for gpytoolbox
        """
        if Os.shape[0] != Ds.shape[0]:
            if Ds.shape[0] == 1:
                self.Os = np.copy(Os)
                self.Ds = np.copy(Os)
                self.Ds[:, :] = Ds[:, :]
            if Os.shape[0] == 1:
                self.Ds = np.copy(Ds)
                self.Os = np.copy(Ds)
                self.Os[:, :] = Os[:, :]
        else:
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

    def __init__(self, r, c, brdf_params=np.array([0, 0, 0, 1]), Le=np.array([0, 0, 0])):
        """
        Initializes a sphere object with its radius, position and albedo.
        """
        self.r = np.float64(r)
        self.c = np.copy(c)
        self.brdf_params = brdf_params
        self.Le = Le
        super().__init__()

    def intersect(self, rays):
        """
        Intersect the sphere with a bundle of rays, and compute the
        distance between the hit point on the sphere surface and the
        ray origins. If a ray did not intersect the sphere, set the
        distance to np.inf.
        """
        distances = np.zeros((rays.Os.shape[0],), dtype=np.float64)
        distances[:] = np.inf
        normals = np.zeros(rays.Os.shape, dtype=np.float64)
        normals[:,:] = np.array([np.inf, np.inf, np.inf])

        # Copy solution from A1
        ### BEGIN SOLUTION
        assert False, "Remove assert and complete this section."
        ### END SOLUTION

        return distances, normals

# triangle mesh objects for our scene
class Mesh(Geometry):
    def __init__(self, filename, brdf_params, Le=np.array([0, 0, 0])):
        """
        Initializes a mesh object with filename of mesh and brdf parameters.
        """
        mesh_blob = np.load(filename)
        
        # mesh vertices, triangles/faces, vertex-normals
        self.v = mesh_blob["v"]
        self.f = mesh_blob["f"]
        self.vn = mesh_blob["vn"]

        self.brdf_params = brdf_params
        self.Le = Le
        super().__init__()  
        
        # ray-traingle interesection precision
        self.rt_precision = np.float32

        # specify maximum memory used for ray-triangle intersection (in GBs)
        self.max_rt_memory = 2

        self.precompute()

    def precompute(self):
        """
        Precomputes quantities useful for ray-triangle intersection.
        """
        tri = self.v[self.f]
        N = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        tri_area = np.linalg.norm(N, axis=1)
        # triangle face-normal
        N /= tri_area[:, np.newaxis]

        # triangle face-tangent
        T = tri[:, 1] - tri[:, 0]
        T /= np.linalg.norm(T, axis=1)[:, np.newaxis]

        # projection matrix to project any point on the triangle plane
        proj_mat = np.stack((T, np.cross(T, N)), axis=1)
        
        # project all triangle vertices to this plane
        tri_p = np.sum(tri[:, :, np.newaxis] * proj_mat[:, np.newaxis], axis=3)
        tri_area_p = Mesh.area2d(tri_p[np.newaxis, :, 1] - tri_p[np.newaxis, :, 0],  tri_p[np.newaxis, :, 2] - tri_p[np.newaxis, :, 0])[0]

        # change precision
        self.tri = tri.astype(self.rt_precision)
        self.N = N.astype(self.rt_precision)
        self.tri_area = tri_area.astype(self.rt_precision)
        self.proj_mat = proj_mat.astype(self.rt_precision)
        self.tri_p = tri_p.astype(self.rt_precision)
        self.tri_area_p = tri_area_p.astype(self.rt_precision)

        # max number of rays in a ray-bundle
        self.rt_batch_size = self.find_ray_batch_size()

    def find_ray_batch_size(self):
        """
        Computes the maximum number rays for processing in a 
        batch to cap memory utilization.
        """
        n_tris = self.tri.shape[0]
        dtype_size = float(np.finfo(self.rt_precision).bits) / 8
        k = 10
        max_bytes = self.max_rt_memory # in GBs
        round_size = 1024

        batch_size = max_bytes * (2**30) / (k * n_tris * dtype_size)
        
        if batch_size < round_size:
            batch_size = round_size
        else:
            batch_size = np.round(batch_size / round_size).astype(int) * round_size

        return batch_size
            
    def mesh_intersect_batch(self, origin, dir):
        """
        Splits the ray bundle into batches for ray-triangle
        intersection.
        """
        origin = origin.astype(self.rt_precision)
        dir = dir.astype(self.rt_precision)
        
        n_rays = dir.shape[0]
       
        if n_rays <= self.rt_batch_size:
            return self.mesh_intersect(origin, dir)

        min_val = np.zeros((n_rays,))
        hit_id = np.zeros((n_rays,), dtype=np.int64)
        barys = np.zeros((n_rays, 3))

        for batch_idx in range(0, n_rays, self.rt_batch_size):
            begin = batch_idx
            end = np.minimum(begin + self.rt_batch_size, n_rays)
            batch_min_val, batch_hit_id, batch_barys = self.mesh_intersect(origin[begin:end], dir[begin:end])

            min_val[begin:end] = batch_min_val
            hit_id[begin:end] = batch_hit_id
            barys[begin:end] = batch_barys
        
        return min_val, hit_id, barys

    @staticmethod
    def area(t0, t1, t2):
        """
        Computes the area of a triangle with three vertices
        as input.
        """
        n = np.cross(t1 - t0, t2 - t0, axis=1)
        return np.linalg.norm(n, axis=1)

    @staticmethod
    def area2d(a, b):
        """
        Computes the cross product of two 2D-vectors.
        """
        return np.abs(a[:,:,0] * b[:,:,1] - a[:,:,1] * b[:,:,0])

    @staticmethod
    def area3d(a, b):
        """
        Computes length of the cross product of two 3D-vectors.
        """
        n = np.cross(a, b, axis=2)
        return np.linalg.norm(n, axis=2)

    @staticmethod
    def get_bary_coords(intersection, tri):
        """
        Compute barycentric coordinates for a list of intersections.
        """
        denom = Mesh.area(tri[:, 0], tri[:, 1], tri[:, 2])
        infMask = np.isinf(intersection)
        intersectionCopy = intersection.copy()
        intersectionCopy[infMask] = -1
        alpha_numerator = Mesh.area(intersectionCopy, tri[:, 1], tri[:, 2])
        beta_numerator = Mesh.area(intersectionCopy, tri[:, 0], tri[:, 2])
        alpha = alpha_numerator / denom
        beta = beta_numerator / denom
        gamma = 1 - alpha - beta
        barys = np.vstack((alpha, beta, gamma)).transpose()
        return barys

    def mesh_intersect(self, origin, dir):
        """
        Compute ray-triangle intersections.
        """
        assert origin.dtype == self.rt_precision
        assert dir.dtype == self.rt_precision
        npMax = np.finfo(self.rt_precision).max

        ## ray plane intersection
        tri_sub_o = self.tri[np.newaxis, :, 0] - origin[:, np.newaxis]
        tri_sub_o_dot_n = np.sum(tri_sub_o * self.N[np.newaxis, :], axis=2)
        dir_dot_n = np.sum(dir[:, np.newaxis] * self.N[np.newaxis, :], axis=2)
        dir_dot_n = np.where(np.isclose(dir_dot_n, 0), 1e-8, dir_dot_n)
        t = tri_sub_o_dot_n / dir_dot_n
        x = origin[:, np.newaxis] + t[:,:,np.newaxis] * dir[:, np.newaxis]
        
        ## Check if the interesction point lies within a triangle
            
        # project all points on triangle's 2D plane
        x_p = np.sum(x[:, :, np.newaxis] * self.proj_mat[np.newaxis, :], axis=3)
        edge = x_p[:,:,np.newaxis] - self.tri_p
        alpha = Mesh.area2d(edge[:,:,1], edge[:,:,2])
        beta = Mesh.area2d(edge[:,:,0], edge[:,:,2])
        gamma = Mesh.area2d(edge[:,:,0], edge[:,:,1])
        accept = np.isclose(alpha + beta + gamma, self.tri_area_p[np.newaxis,:])
        reject = np.logical_not(accept)
        reject = np.logical_or(reject, t < 0)
        accept = np.logical_not(reject)
        t = t * accept + npMax * reject 
       
        # Find neareast intersection
        hit_id = np.argmin(t, axis=1)
        min_val = np.min(t, axis=1)
        #min_val = t[np.arange(hitId.shape[0]), hitId] ## Apparently slower
        min_val[min_val == npMax] = np.Inf
        hit_id[min_val == np.Inf] = -1

        # Find barycentric coordinates
        intersection = origin + min_val[:, None] * dir
        tri_hit = self.tri[hit_id]
        barys = Mesh.get_bary_coords(intersection, tri_hit)
       
        return min_val.astype(np.float64), hit_id, barys.astype(np.float64)

    def intersect(self, rays):
        hit_normals = np.array([np.inf, np.inf, np.inf])
        hit_distances, triangle_hit_ids, barys = self.mesh_intersect_batch(rays.Os, rays.Ds)
        
        # [TODO] Following line compute face normals,
        # replace the next line with your code for Phong normal interpolation
        ### BEGIN SOLUTION
        temp_normals = self.N[triangle_hit_ids]
        ### END SOLUTION

        temp_normals = np.where((triangle_hit_ids == -1)[:, np.newaxis],
                                hit_normals,
                                temp_normals)
        hit_normals = temp_normals

        return hit_distances, hit_normals

# Enumerate the different importance sampling strategies we will implement
UNIFORM_SAMPLING, COSINE_SAMPLING = range(2)

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

    def generate_eye_rays(self, jitter=False):
        """
        Generate a bundle of eye rays.

        The eye rays originate from the eye location, and shoots through each
        pixel into the scene.
        """
        # Copy solution from A1, and
        # [TODO] Add jitter to each pixel
        ### BEGIN SOLUTION
        assert False, "Remove assert and complete this section."
        ### END SOLUTION

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """
        hit_ids = np.array([-1])
        hit_distances = np.array([np.inf])
        hit_normals = np.array([np.inf, np.inf, np.inf])

        # [TODO] Iterate over all geometries and find nearest hit information
        ### BEGIN SOLUTION
        assert False, "Remove assert and complete this section."
        ### END SOLUTION

        return hit_distances, hit_normals, hit_ids

    def render(self, eye_rays, sampling_type=UNIFORM_SAMPLING, num_samples=1):
        # vectorized scene intersection
        shadow_ray_o_offset = 1e-6
        distances, normals, ids = self.intersect(eye_rays)

        normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
                           normals, np.array([0, 0, 0]))

        hit_points = eye_rays(distances)

        # CAREFUL! when ids == -1 (i.e., no hit), you still get a valid BRDF value!
        # taking brdf_params[:-1] because for now we're only dealing with diffuse BRDFs
        brdf_params = np.array([obj.brdf_params[:-1] for obj in self.geometries])[ids]

        # initialize the output image
        L = np.zeros(normals.shape, dtype=np.float64)

        # base code renders out normals; useful for the first two deliverables
        if len(self.lights) == 0:
            L = np.abs(normals)

        # [TODO] replace these lines with your solution for deliverables 3 (and 4, for ECSE 546 students)
        for l in range(len(self.lights)):
            curr_light = self.lights[l]
            # Uniform light background color
            L = np.where(normals != np.array([0, 0, 0]), np.array([0, 0, 0]), curr_light["color"])
            # Ambient Occlusion integrator
            ### BEGIN SOLUTION
            if sampling_type == UNIFORM_SAMPLING:
                # [TODO] Populate L with uniform samples with an average over num_samples
                pass
            elif sampling_type == COSINE_SAMPLING:
                # [TODO] For 556 Only, 
                # Populate L with cosine importance samples with an average over num_samples
                pass
            ### END SOLUTION
       
        L = L.reshape((self.h, self.w, 3))
        return L

    def progressive_render_display(self, jitter=False, total_spp=20, spppp=1,
                                   sampling_type=UNIFORM_SAMPLING):
        # matplotlib voodoo to support redrawing on the canvas
        plt.figure()
        plt.ion()
        plt.show()

        L = np.zeros((self.h, self.w, 3), dtype=np.float64)

        # more matplotlib voodoo: update the plot using the 
        # image handle instead of looped imshow for performance
        image_data = plt.imshow(L)

        # number of rendering iterations needed to obtain 
        # (at least) our total desired spp
        progressive_iters = int(np.ceil(total_spp / spppp))

        ### BEGIN SOLUTION
        # [TODO] replace the next five lines with 
        # your progressive rendering display loop
        for i in range(progressive_iters):
            vectorized_eye_rays = self.generate_eye_rays(jitter)
            plt.title(f"current spp: {(i + 1) * spppp} of {progressive_iters * spppp}")
            L = self.render(vectorized_eye_rays, sampling_type, spppp)
            image_data.set_data(np.clip(L, 0, 1))
            plt.pause(0.001) # add a tiny delay between rendering passes
        ### END SOLUTION

        plt.savefig(f"render-{progressive_iters * spppp}spp.png")
        plt.show(block=True)

if __name__ == "__main__":
    enabled_tests = [True, True, True, True]
    test_bunny = False

    ground = Sphere(1000, np.array([0, -1002.5, 0]),
                        brdf_params=np.array([0.9, 0.9, 0.9, 1.0]))
    analytic_sphere = Sphere(1.5, np.array([-1.75, -0.5, -0.0]),
                        brdf_params=np.array([0.9, 0.9, 0.9, 1.0]))
    mesh_sphere = Mesh("lp_sphere.npz",
                        brdf_params=np.array([0.9, 0.9, 0.9, 1.0]))
    mesh_bunny = Mesh("bunny-446rt.npz",
                        brdf_params=np.array([0.9, 0.9, 0.9, 1.0]))

    #########################################################################
    ### Deliverable 1 TESTS Eye Ray Anti Aliasing and Progressive Rendering
    #########################################################################
    if enabled_tests[0]:
        # Create test scene and test sphere
        scene = Scene(w=int(1024 / 4), h=int(768 / 4))  # DEBUG: use a lower resolution to debug
        scene.set_camera_parameters(
            eye=np.array([0, 0.5, -5], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        if test_bunny:
            scene.add_geometries([ground, mesh_bunny])
        else:
            scene.add_geometries([ground, analytic_sphere, mesh_sphere])

        # no-AA
        scene.progressive_render_display(jitter=False)

        # with AA
        scene.progressive_render_display(jitter=True)

    #########################################################################
    ### Deliverable 2 TESTS Mesh Intersection and Phong Normal Interpolation
    #########################################################################
    if enabled_tests[1]:
        # Create test scene and test sphere
        scene = Scene(w=int(1024 / 4), h=int(768 / 4))  # DEBUG: use a lower resolution to debug
        scene.set_camera_parameters(
            eye=np.array([0, 0.5, -5], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        if test_bunny:
            scene.add_geometries([ground, mesh_bunny])
        else:
            scene.add_geometries([ground, analytic_sphere, mesh_sphere])
        
        # no-AA
        scene.progressive_render_display(jitter=False, total_spp=1, spppp=1)

    ###########################################################################
    ### Deliverable 3 TESTS Ambient Occlusion with Uniform Importance Sampling
    ###########################################################################
    if enabled_tests[2]:
        # Create test scene and test sphere
        scene = Scene(w=int(1024 / 4), h=int(768 / 4))  # DEBUG: use a lower resolution to debug
        scene.set_camera_parameters(
            eye=np.array([0, 0.5, -5], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        if test_bunny:
            scene.add_geometries([ground, mesh_bunny])
        else:
            scene.add_geometries([ground, analytic_sphere, mesh_sphere])

        scene.add_lights([
            {
                "type": "uniform",
                "color": np.array([0.9, 0.9, 0.9])
            }
        ])

        scene.progressive_render_display(jitter=True, total_spp=100, spppp=2,
                                         sampling_type=UNIFORM_SAMPLING)

    #########################################################################################
    ### ECSE 546 Only: Deliverable 4 TESTS Ambient Occlusion with Cosine Importance Sampling
    #########################################################################################
    if enabled_tests[3]:
        # Create test scene and test sphere
        scene = Scene(w=int(1024 / 4), h=int(768 / 4))  # DEBUG: use a lower resolution to debug
        scene.set_camera_parameters(
            eye=np.array([0, 0.5, -5], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        if test_bunny:
            scene.add_geometries([ground, mesh_bunny])
        else:
            scene.add_geometries([ground, analytic_sphere, mesh_sphere])

        scene.add_lights([
            {
                "type": "uniform",
                "color": np.array([0.9, 0.9, 0.9])
            }
        ])

        scene.progressive_render_display(jitter=True, total_spp=100, spppp=2,
                                         sampling_type=COSINE_SAMPLING)
