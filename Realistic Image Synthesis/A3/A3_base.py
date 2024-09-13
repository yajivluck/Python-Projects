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
    #4x4 matrix for inverse look at transform
    matrix = np.zeros((4,4))
    
    z_c = normalize(at-eye)
    #normalize up in case it is not

    x_c = normalize(np.cross(up,z_c))
    y_c = normalize(np.cross(z_c,x_c))
    
    matrix[0:3, 0] = x_c
    matrix[0:3, 1] = y_c
    matrix[0:3, 2] = z_c
    matrix[0:3,3] = eye
    matrix[3,3] = 1
    
    return np.linalg.inv(matrix)
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
        # Remove if required
        distances = np.zeros((rays.Os.shape[0],), dtype=np.float64)
        distances[:] = np.inf
        normals = np.zeros(rays.Os.shape, dtype=np.float64)
        normals[:,:] = np.array([np.inf, np.inf, np.inf])

        # Copy solution from A2
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
        
        # # Copy solution from A2
        # [TODO] Following line compute face normals,
        # replace the next line with your code for Phong normal interpolation
        ### BEGIN SOLUTION
        # Vertex normals for triangles hit by the rays
        normals_for_hits = self.vn[self.f[triangle_hit_ids]]
        # Compute Phong interpolated normals
        temp_normals = np.sum(normals_for_hits * barys[:, :, np.newaxis], axis=1)
        # Normalize the interpolated normals
        temp_normals /= np.linalg.norm(temp_normals, axis=1, keepdims=True)
        # Set normals to [inf, inf, inf] for rays that didn't hit any triangle
        temp_normals[triangle_hit_ids == -1] = hit_normals
        
        ##Default Normals
        #temp_normals = self.N[triangle_hit_ids]
        

        temp_normals = np.where((triangle_hit_ids == -1)[:, np.newaxis],
                                hit_normals,
                                temp_normals)
        hit_normals = temp_normals
        ### END SOLUTION

        return hit_distances, hit_normals

# Enumerate the different importance sampling strategies we will implement
UNIFORM_SAMPLING, LIGHT_SAMPLING, BRDF_SAMPLING, MIS_SAMPLING = range(4)

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
        """ 
        Adds a list of geometries to the scene.
        
        For geometries with non-zero emission,
        additionally add them to the light list.
        """
        for i in range(len(geometries)):
            if (geometries[i].Le != np.array([0, 0, 0])).any():
                self.add_lights([geometries[i]])

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
        # Copy solution from A2
        ### BEGIN SOLUTION
        w,h = self.w,self.h
        
        x_step,y_step = 2.0/w, 2.0/h
         
        x_coords = np.linspace(-1,1,w,endpoint = False) + (x_step / 2)
        #Invert y coords to make meshgrid match with out coords system in NDC
        y_coords = np.linspace(1, -1, h, endpoint = False) - (y_step / 2)
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
        
        #instead of centering eye rays, randomly pick over area of pixel
        if jitter:

            #multiply mesh coords by random fraction of x_step/y_step.
            #Then substracting by half that step so that the values
            #of x_jitter/y_jitter lies between -x_step/2 and x_step/2 or -y_step/2 and y_step/2
            #Add those random displacements to the original centered NDC to "jitter them"
            
            x_jitter, y_jitter = x_step * np.random.rand(self.h,self.w) - (x_step/2),  y_step * np.random.rand(self.h,self.w) - (y_step/2), 
            x_mesh += x_jitter
            y_mesh += y_jitter
            
        #height fov convert degree to randians and take tan to get the scale from NDC->Camera
        height_fov = np.tan(np.radians(self.fov/2))  
        #Width fov as aspect ratio times height fov
        width_fov = height_fov * w / h
        
        
        cam_x,cam_y = x_mesh * width_fov, y_mesh * height_fov
        #Cam_z is shape of cam_x filled by 1 (z_c = 1)
        cam_z = np.full(cam_x.shape,1)
        
        #Build matrix of zeros of shape of cam_x expanded with 3 element vectors for each eye ray)
        cam_coords = np.zeros((cam_x.shape[0], cam_x.shape[1],3))
        #Set cam coords with individual component of inner eye ray vector
        cam_coords[:,:,0] = cam_x
        cam_coords[:,:,1] = cam_y
        cam_coords[:,:,2] = cam_z

        #Flatten cam coords as list of eye ray vector instead of 2d matrix of vectors
        cam_coords = cam_coords.reshape(-1,3)
        
        #Add 4th dimension zero to fit dimensions of dot product with inverse looktat matrix
        fourth_dim_zeros = np.zeros((cam_coords.shape[0],1))
        
        #Add fourth dim zeros
        cam_coords = np.hstack((cam_coords, fourth_dim_zeros))
        
        #Convert to world space using look_at_inv
        world_space_coords = np.dot(cam_coords, look_at_inv(eye = self.eye, at = self.at, up = self.up))
        
        #Normalize world space coords and only take first three element of inner most 4 dim vectors (we don't care about last element)
        world_space_coords = np.apply_along_axis(normalize, axis = 1, arr = world_space_coords[:,:3])
        
                
        Bundle_of_rays = Rays(Os = np.asarray([self.eye]), Ds = world_space_coords)
        return Bundle_of_rays
        ### END SOLUTION

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """
        # remove if required
        hit_ids = np.array([-1])
        hit_distances = np.array([np.inf])
        hit_normals = np.array([np.inf, np.inf, np.inf])

        # copy from A2
        # [TODO] Iterate over all geometries and find nearest hit information
        ### BEGIN SOLUTION
        assert False, "Remove assert and complete this section."
        ### END SOLUTION

        return hit_distances, hit_normals, hit_ids

    def render(self, eye_rays, sampling_type=UNIFORM_SAMPLING):
        # vectorized scene intersection
        shadow_ray_o_offset = 3e-6 # Warning: Do not modify
        distances, normals, ids = self.intersect(eye_rays)

        normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
                           normals, np.array([0, 0, 0]))

        hit_points = eye_rays(distances)

        # NOTE: When ids == -1 (i.e., no hit), you get a valid BRDF ([0,0,0,0]), L_e ([0,0,0]), and objects id (-1)!
        brdf_params = np.concatenate((np.array([obj.brdf_params for obj in self.geometries]),
                                      np.array([0, 0, 0, 1])[np.newaxis, :]))[ids]
        L_e = np.concatenate((np.array([obj.Le for obj in self.geometries]),
                              np.array([0, 0, 0])[np.newaxis, :]))[ids]
        objects = np.concatenate((np.array([obj for obj in self.geometries]),
                                  np.array([-1])))
        hit_objects = np.concatenate((np.array([obj for obj in self.geometries]),
                                      np.array([-1])))[ids]

        # initialize the output image
        L = np.zeros(normals.shape, dtype=np.float64)

        ### BEGIN SOLUTION

        # Directly render light sources
        L = np.where(np.logical_and(L_e != np.array([0, 0, 0]), (ids != -1)[:, np.newaxis]), L_e, L)

        for l in range(len(self.lights)):
            if sampling_type == UNIFORM_SAMPLING:
                # uniform spherical samples
                pass
            elif sampling_type == LIGHT_SAMPLING:
                # uniform spherical solid angle sampling
                pass
            elif sampling_type == BRDF_SAMPLING:
                # cosine-power solid angle sampling
                pass
            elif sampling_type == MIS_SAMPLING:
                # MIS sampling
                pass               
        ### END SOLUTION

        L = L.reshape((self.h, self.w, 3))
        return L

    def progressive_render_display(self, jitter=False, total_spp=20,
                                   sampling_type=UNIFORM_SAMPLING):
        suffix = ["uniform", "light", "brdf", "mis"]
        plt.figure()
        plt.ion()
        plt.show()

        L = np.zeros((self.h, self.w, 3), dtype=np.float64)
        # Update the plot using the image handle instead of looped imshow for performance
        image_artist = plt.imshow(L)
        
        # [TODO] replace the next lines with your progressive rendering display loop
        ### BEGIN SOLUTION
        for i in range(total_spp):
            plt.title(f"current spp: {(i + 1)} of {total_spp}")
            assert False, "Remove assert and complete this section."
            plt.pause(0.001)
        ### END SOLUTION

        plt.savefig(f"render-{suffix[sampling_type]}-{total_spp}spp.png", dpi=200)
        plt.show(block=True)

if __name__ == "__main__":
    enabled_tests = [True, True, True]

    # Create test scene and test sphere
    scene = Scene(w=int(512), h=int(512))  # TODO: debug at lower resolution
    scene.set_camera_parameters(
        eye=np.array([0, 2, 15], dtype=np.float64),
        at=normalize(np.array([0, -2, 2.5], dtype=np.float64)),
        up=np.array([0, 1, 0], dtype=np.float64),
        fov=int(40)
    )

    # Veach Scene Lights
    scene.add_geometries([
        Sphere(0.0333, np.array([3.75, 0, 0]),
               Le=10 * np.array([901.803, 0, 0])),
        Sphere(0.1, np.array([1.25, 0, 0]),
               Le=10 * np.array([0, 100, 0])),
        Sphere(0.3, np.array([-1.25, 0, 0]),
               Le=10 * np.array([0, 0, 11.1111])),
        Sphere(0.9, np.array([-3.75, 0, 0]),
               Le=10 * np.array([1.23457, 1.23457, 1.23457])),
        Sphere(0.5, np.array([-10, 10, 4]),
               Le=np.array([800, 800, 800]))
    ])

    # Geometry
    scene.add_geometries([
        Mesh("plate1.npz", brdf_params=np.array([1, 1, 1, 30000])),
        Mesh("plate2.npz", brdf_params=np.array([1, 1, 1, 5000])),
        Mesh("plate3.npz", brdf_params=np.array([1, 1, 1, 1500])),
        Mesh("plate4.npz", brdf_params=np.array([1, 1, 1, 100])),
        Mesh("floor.npz", brdf_params=np.array([0.5, 0.5, 0.5, 1]))
    ])

    #########################################################################
    ### Deliverable 1 TEST: comment/modify as you see fit
    #########################################################################
    if enabled_tests[0]:
        scene.progressive_render_display(total_spp=1, jitter=True, sampling_type=LIGHT_SAMPLING)
        scene.progressive_render_display(total_spp=10, jitter=True, sampling_type=LIGHT_SAMPLING)
        scene.progressive_render_display(total_spp=100, jitter=True, sampling_type=LIGHT_SAMPLING)

    #########################################################################
    ### Deliverable 2 TEST: comment/modify as you see fit
    #########################################################################
    if enabled_tests[1]:
        scene.progressive_render_display(total_spp=1, jitter=True, sampling_type=BRDF_SAMPLING)
        scene.progressive_render_display(total_spp=10, jitter=True, sampling_type=BRDF_SAMPLING)
        scene.progressive_render_display(total_spp=100, jitter=True, sampling_type=BRDF_SAMPLING)

    #########################################################################
    ### Deliverable 3 TEST (Only for ECSE 546 students!): comment/modify as you see fit
    #########################################################################
    if enabled_tests[2]:
        scene.progressive_render_display(total_spp=1, jitter=True, sampling_type=MIS_SAMPLING)
        scene.progressive_render_display(total_spp=10, jitter=True, sampling_type=MIS_SAMPLING)
        scene.progressive_render_display(total_spp=100, jitter=True, sampling_type=MIS_SAMPLING)
