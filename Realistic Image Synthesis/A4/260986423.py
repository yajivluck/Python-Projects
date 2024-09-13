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

        # Copy solution from A2/A3
        ### BEGIN SOLUTION
        # Remove if required
        distances = np.zeros((rays.Os.shape[0],), dtype=np.float64)
        distances[:] = np.inf
        normals = np.zeros(rays.Os.shape, dtype=np.float64)
        normals[:,:] = np.array([np.inf, np.inf, np.inf])

        # Copy solution from A2
        ### BEGIN SOLUTION
        #Inner function to get distance from intersecting point
        def get_distance():
            # Define coefficient for quadratic to solve
            A = 1
            B = 2 * np.sum((rays.Os - self.c) * rays.Ds, axis=1)
            C = np.sum((rays.Os - self.c)**2, axis=1) - self.r**2
            
            # Calculate the discriminants
            discriminants = B**2 - 4*A*C
                        
            # Mask for zero discriminants based on epsilon
            zero = discriminants == 0
            
            # For discriminants close to zero, the solution is easier to compute
            distances[zero] = -B[zero] / (2 * A)
            
            # Mask for positive discriminants
            positive_discriminant = discriminants > 0
            
            # Calculate t1 and t2
            t1 = (-B[positive_discriminant] - np.sqrt(discriminants[positive_discriminant])) / (2 * A)
            t2 = (-B[positive_discriminant] + np.sqrt(discriminants[positive_discriminant])) / (2 * A)
            
            # Create masks for non-negative t1 and t2
            non_negative_t1 = t1 >= 0
            non_negative_t2 = t2 >= 0
            
            # Use np.where to classify the combination of t solns (based on whether they are positive or negative)
            distances[positive_discriminant] = np.where(
                non_negative_t1 & non_negative_t2, np.minimum(t1, t2),      # both t1 and t2 non-negative
                np.where(non_negative_t1, t1,                               # only t1 is non-negative
                         np.where(non_negative_t2, t2, np.inf))             # only t2 is non-negative
            )
            
            
            # Return only the non-negative values of t (greater than EPSILON_SPHERE )
            distances[distances <= 0] = np.inf
            return distances
        
        #Get hit distance from inner function
        distances = get_distance()
        
        #bool mask of valid hit distances
        valid_hits = distances < np.inf
                
        # Compute the intersection points for the intersected rays
        intersection_points = rays(distances)

        #Calculate hit normals for intersected rays
        if self.r != 0:
            normals[valid_hits] = (intersection_points[valid_hits] - self.c) / self.r
    
        else:
            normals[valid_hits] =  (intersection_points[valid_hits] - self.c)

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
        
        # # Copy solution from A2/A3
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
IMPLICIT_UNIFORM_SAMPLING, EXPLICIT_UNIFORM_SAMPLING, IMPLICIT_BRDF_SAMPLING, EXPLICIT_LIGHT_BRDF_SAMPLING = range(4)

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
        # Copy solution from A2/A3
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

        # copy from A2/A3
        # [TODO] Iterate over all geometries and find nearest hit information
        ### BEGIN SOLUTION
        hit_distances = []
        hit_normals = []

        #Enumerate for loop for hit_id and sphere
        for hit_id, shape in enumerate(self.geometries):
            #Get the times (as distance) that the geomerty was hit by the rays
            hit_distance, hit_normal = shape.intersect(rays)
            hit_distances.append(hit_distance)
            hit_normals.append(hit_normal)

        # Convert the list of arrays into a 2D numpy array, zipped together
        zipped_distances = np.column_stack(hit_distances)
        # Convert the list of normals into a 2D numpy array, zipped together

        hit_normals = np.stack(hit_normals)
    
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
        
        
        hit_normals = hit_normals[hit_ids, np.arange(hit_ids.size)]
        hit_normals[~id_mask] = np.array([np.inf,np.inf,np.inf])
        
        # Ensure no inf warnings in hit_normals
        hit_normals = np.where(hit_normals == np.inf, 0, hit_normals)
        ### END SOLUTION
        return hit_distances, hit_normals, hit_ids

        

    def render(self, eye_rays, num_bounces=3, sampling_type=IMPLICIT_BRDF_SAMPLING):
        # vectorized scene intersection
        shadow_ray_o_offset = 1e-4 # Warning: Do not modify
        distances, normals, ids = self.intersect(eye_rays)

        hit_normals = np.where(np.isfinite(normals),
                               normals, np.array([0, 0, 0]))

        hit_points = eye_rays(distances)

        # NOTE: When ids == -1 (i.e., no hit), you get a valid BRDF ([0,0,0,0]), L_e ([0,0,0]), and objects id (-1)!
        brdf_params = np.concatenate((np.array([obj.brdf_params for obj in self.geometries]),
                                      np.array([0, 0, 0, 1])[np.newaxis, :]))[ids]
        L_e = np.concatenate((np.array([obj.Le for obj in self.geometries]),
                              np.array([0, 0, 0])[np.newaxis, :]))[ids]
        objects = np.concatenate((np.array([obj for obj in self.geometries]),
                                  np.array([-1])))

        # initialize the output image
        L = np.zeros(hit_normals.shape, dtype=np.float64)

        # Directly render light sources
        L = np.where(np.logical_and(L_e != np.array([0, 0, 0]), (ids != -1)[:, np.newaxis]), L_e, L)

        ### BEGIN SOLUTION
        # PLACEHOLDER: our base code renders out debug normals.
        # [TODO] Replace these next three lines with your
        # solution for your deliverables

        #For Deliverable 1
        if sampling_type == IMPLICIT_BRDF_SAMPLING:
                        
    
            # Initialize the throughput and active rays
            throughput = np.ones_like(hit_normals)
            #Instantiate hit_mask to only consider points where eye_rays originally intersected with
            hit_mask = ids != -1
                        
            for bounce in range(num_bounces):
                

                #This makes use of the hit_points and normals from eye_rays first, but those are updated at every bounce
                          
                # Only continue if there are active paths
                if not np.any(hit_mask):
                    break
                        
                #Get brdf params
                pd = brdf_params[:, 0:3]
                alpha = brdf_params[:, 3]
                #Mask for diffuse surfaces
                diffuse_mask = alpha == 1
                 
                #Origins of shadow rays, offsetted by the shadow_ray_o_offset in the normal direction
                Os = hit_points + shadow_ray_o_offset * normals       
                
                # Generate random samples for each grid element and for each sample
                xi1 = np.random.rand(normals.shape[0], 1)
                xi2 = np.random.rand(normals.shape[0], 1)
            
                # Calculate ray directions (canonic)         
                omega_z = xi1 ** (1 / (alpha.reshape(-1, 1) + 1))
                r = np.sqrt(np.clip(1 - omega_z**2, 0, 1))
                                
                #phi between 0 and 2pi
                phi = 2 * np.pi * xi2
                omega_x = r * np.cos(phi)
                omega_y = r * np.sin(phi)
              
                # Stack the omegas
                omegas = np.stack((omega_x, omega_y, omega_z), axis=-1)
                omegas = omegas.reshape(omegas.shape[0],3)
                
                #Make space for transformed omegas
                transformed_omegas = np.zeros_like(omegas)
                
                
                if np.any(~diffuse_mask):
                    
                    
                    #Get wo for non diffuse surfaces (opposite of direction of eye_rays)                
                    wo = -eye_rays.Ds[~diffuse_mask] 
                    
                    # Compute ωr for each hit point where alpha is not zero
                    omega_r = 2 * np.sum(normals[~diffuse_mask] * wo, axis=1, keepdims=True) * normals[~diffuse_mask] - wo
                
                    #Build new orthonormal basis for specular surface
                    
                    #x-axis of new coordinate system cross omega_r with y
                    u = np.cross(omega_r, np.array([0,1,0]))
                    
                    # If omega_r is nearly parallel to the up vector, use a x-axis instead (doesn't matter)
                    #This is to prevent very small number as the vector for the basis
                    u_norm = np.linalg.norm(u, axis=1, keepdims=True)
                    #If u_norm is too small, use x-axis instead to compute it
                    close_to_parallel_wc = np.squeeze(u_norm < 1e-5)
                    u[close_to_parallel_wc] = np.cross(omega_r[close_to_parallel_wc], np.array([1, 0, 0]))
                    #Normalize u vectors
                    u = np.apply_along_axis(normalize, axis=1, arr=u)
                    
                    #Compute other vector in orthonormal basis using omega_r and u 
                    v = np.cross(omega_r, u)
                    v = np.apply_along_axis(normalize, axis=1, arr=v)
                    
                    #Build transformation matrix to coordinate around the cone for each light sphere
                    T_specular = np.stack((u,v,omega_r), axis=-2)
                    
                    #Transform specular omegas by rotating them along the respective omega_r
                    transformed_omegas_specular = np.einsum('ij,ijk->ik', omegas[~diffuse_mask], T_specular)
                    #Fill transformed_omegas appropriately
                    transformed_omegas[~diffuse_mask] = transformed_omegas_specular
                    
                    #Dot_product between normals and transformed_omegas
                    Dot_product = np.sum(transformed_omegas[~diffuse_mask] * normals[~diffuse_mask], axis=1, keepdims=True)
            
                    #Set to 0 negative dot products
                    Dot_product = np.maximum(0,Dot_product)
                    
                    throughput[hit_mask][~diffuse_mask] *= Dot_product[hit_mask].squeeze().reshape(-1, 1)
                    
                
                if np.any(diffuse_mask):
                    #Rotate along the normal at spec
                    
                    #x-axis of new coordinate system cross the normal vector with the x-axis
                    u_n = np.cross(normals[diffuse_mask], np.array([0,1,0]))
                    
                    # If wc is nearly parallel to the up vector, use a x-axis instead (doesn't matter)
                    #This is to prevent very small number as the vector for the basis
                    u_n_norm = np.linalg.norm(u_n, axis=1, keepdims=True)
                    #If u_norm is too small, use x-axis instead to compute it
                    close_to_parallel_n = np.squeeze(u_n_norm < 1e-5)
                    u_n[close_to_parallel_n] = np.cross(normals[diffuse_mask][close_to_parallel_n], np.array([1, 0, 0]))
                    #Normalize u vectors
                    u_n = np.apply_along_axis(normalize, axis=1, arr=u_n)
                    
                    #Compute other vector in orthonormal basis using wc and u 
                    v_n = np.cross(normals[diffuse_mask], u_n)
                    v_n = np.apply_along_axis(normalize, axis=1, arr=v_n)
                    
                    #Build transformation matrix to coordinate around the cone for each light sphere
                    T_diffuse = np.stack((u_n,v_n,normals[diffuse_mask]), axis=-2)
                    
                    #Transform random ray by the appropriate rotation (along hit normal) if it hit a diffuse surface
                    transformed_omegas_diffuse = np.einsum('ij,ijk->ik', omegas[diffuse_mask], T_diffuse)
                    
                    #Build transformed omegas with respective corresponding random rays
                    transformed_omegas[diffuse_mask] = transformed_omegas_diffuse
                
                
                
                # Find which vectors are not zero (light hits)
                light_hit = np.any(L_e != 0, axis=1)
                
                #Common design pattern, multiplying throughput by corresponding contribution (diffuse and phong brdf)       
                #Update if no light hit (only surfaces)
                throughput[hit_mask] *= pd[hit_mask]
                
                #If path hits emitter, set as L_e value
                throughput[hit_mask & light_hit] = L_e[hit_mask & light_hit]
                
                #If ray went out of bounds, set contribution to 0
                throughput[hit_mask & ids == -1] = np.array([0,0,0])
                
                #Update hit_mask for next round of bounces by discarding rays that haven't hit anything (exited the scene) or that hit lights in the current bounce
                hit_mask[light_hit] = False
                hit_mask[ids == -1] = False
       
                    
                               
          
                
          
                #Shadow Ray instantiation                       
                eye_rays = Rays(Os = Os, Ds = transformed_omegas)
                #Get shadow ray hit distances
                distances, normals, ids = self.intersect(eye_rays)
                
                         
                
                #TODO brdf params and L_E probably mess with output. 
                
                
                #Update brdf and L_e
                brdf_params = np.concatenate((np.array([obj.brdf_params for obj in self.geometries]),
                                              np.array([0, 0, 0, 1])[np.newaxis, :]))[ids]
                
                L_e = np.concatenate((np.array([obj.Le for obj in self.geometries]),
                                      np.array([0, 0, 0])[np.newaxis, :]))[ids]

                #Update hits/normals for next bounce
                hit_points = eye_rays(distances)
                
         

  
            #If path still active without light, set to 0
            throughput[hit_mask] = np.array([0,0,0])
            
            L = throughput #Add original light source to the rendered image
            #After Adding all bounces contribution, reshape L and return it 
            L = L.reshape((self.h, self.w, 3))
            
        return L
        ### END SOLUTION

    def progressive_render_display(self, jitter=False, total_spp=20, num_bounces=3,
                                   sampling_type=IMPLICIT_BRDF_SAMPLING):
        # matplotlib voodoo to support redrawing on the canvas
        plt.figure()
        plt.ion()
        plt.show()

        L = np.zeros((self.h, self.w, 3), dtype=np.float64)

        # more matplotlib voodoo: update the plot using the
        # image handle instead of looped imshow for performance
        image_data = plt.imshow(L)

        ### BEGIN CODE (note: we will not grade your progressive rendering code in A4)
        # [TODO] replace the next five lines with any
        # accumulation, output and/or display code you wish
        vectorized_eye_rays = self.generate_eye_rays(jitter)
        L = self.render(vectorized_eye_rays, num_bounces, sampling_type)
        image_data.set_data(L)
        plt.pause(0.0001)  # add a tiny delay between rendering passes
        ### END CODE
             
        for i in range(total_spp):
            plt.title(f"current spp: {(i + 1)} of {total_spp}")           
            vectorized_eye_rays = self.generate_eye_rays(jitter)
            L = (L + self.render(vectorized_eye_rays, sampling_type)) 
            run_avg = np.clip(L / (i+1), 0, 1)
            plt.imshow(run_avg)
            image_data.set_data(run_avg)
            plt.pause(0.001)
        
        plt.savefig(f"render-{total_spp}spp.png")
        plt.show(block=True)

if __name__ == "__main__":
    enabled_tests = [True, False, False]
    enable_deliverables = [True, False]  # NOTE: 546 students can set the second boolean to True

    #########################################################################
    ### Test Case 1: Default Cornell Box Scene
    #########################################################################
    if enabled_tests[0]:
        # Create test scene and test sphere
        scene = Scene(w=int(512 / 2), h=int(512 / 2))  # TODO: debug at lower resolution
        scene.set_camera_parameters(
            eye=np.array([278, 273, -770], dtype=np.float64),
            at=(np.array([278, 273, -769], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=int(39)
        )

        scene.add_geometries([
            Sphere(60, np.array([213 + 65, 450, 227 + 105 / 2 - 100]),
                   Le=1.25 * np.array([15.6, 15.6, 15.6])),
            Mesh("cbox_floor.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_ceiling.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_back.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_greenwall.npz",
                 brdf_params=np.array([0.16, 0.76, 0.16, 1])),
            Mesh("cbox_redwall.npz",
                 brdf_params=np.array([0.76, 0.16, 0.16, 1])),
            Mesh("cbox_smallbox.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_largebox.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1]))
        ])

        #########################################################################
        ### Deliverable 1: Implicit BRDF Sampling
        #########################################################################
        if enable_deliverables[0]:
            

            scene.progressive_render_display(total_spp=512, jitter=True, num_bounces=2,
                                              sampling_type=IMPLICIT_BRDF_SAMPLING)
            # scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=2,
            #                                 sampling_type=IMPLICIT_BRDF_SAMPLING)
            # scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=3,
            #                                  sampling_type=IMPLICIT_BRDF_SAMPLING)
            # scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=4,
            #                                 sampling_type=IMPLICIT_BRDF_SAMPLING)

        #########################################################################
        ### Deliverable 2: ECSE 546 Only - Explicit Light BRDF Sampling
        #########################################################################
        if enable_deliverables[1]:
            scene.progressive_render_display(total_spp=1, jitter=True, num_bounces=2,
                                            sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=2,
                                            sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

            scene.progressive_render_display(total_spp=1, jitter=True, num_bounces=3,
                                            sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=3,
                                            sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=100, jitter=True, num_bounces=3,
                                            sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

            scene.progressive_render_display(total_spp=1, jitter=True, num_bounces=4,
                                            sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=4,
                                            sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=100, jitter=True, num_bounces=4,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

    #########################################################################
    ### Test Case 2: Scene with decreasing light size (constant power)
    #########################################################################
    if enabled_tests[1]:
        # Create test scene and test sphere
        scene = Scene(w=int(512 / 2), h=int(512 / 2))  # TODO: debug at lower resolution
        scene.set_camera_parameters(
            eye=np.array([278, 273, -770], dtype=np.float64),
            at=(np.array([278, 273, -769], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=int(39)
        )

        scene.add_geometries([
            Sphere(60, np.array([213 + 65, 450, 227 + 105 / 2 - 100]),
                   Le=1.25 * np.array([15.6, 15.6, 15.6])),
            Mesh("cbox_floor.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_ceiling.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_back.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_greenwall.npz",
                 brdf_params=np.array([0.16, 0.76, 0.16, 1])),
            Mesh("cbox_redwall.npz",
                 brdf_params=np.array([0.76, 0.16, 0.16, 1])),
            Mesh("cbox_smallbox.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_largebox.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1]))
        ])

        #########################################################################
        ### Deliverable 1: Implicit BRDF Sampling
        #########################################################################
        if enable_deliverables[0]:
            scene.geometries[0].r = 60
            scene.geometries[0].Le = 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=2,
                                            sampling_type=IMPLICIT_BRDF_SAMPLING)

            scene.geometries[0].r = 30
            scene.geometries[0].Le = 4 * 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=2,
                                            sampling_type=IMPLICIT_BRDF_SAMPLING)

            scene.geometries[0].r = 10
            scene.geometries[0].Le = 9 * 4 * 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=2,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)

        #########################################################################
        ### Deliverable 2: ECSE 546 Only - Explicit Light BRDF Sampling
        #########################################################################
        if enable_deliverables[1]:
            scene.geometries[0].r = 60
            scene.geometries[0].Le = 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=2,
                                            sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

            scene.geometries[0].r = 30
            scene.geometries[0].Le = 4 * 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=2,
                                            sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

            scene.geometries[0].r = 10
            scene.geometries[0].Le = 9 * 4 * 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

    #########################################################################
    ### Test Case 3: Scene with different BRDFs
    #########################################################################
    if enabled_tests[2]:
        # Create test scene and test sphere
        scene = Scene(w=int(512 / 2), h=int(512 / 2))  # TODO: debug at lower resolution
        scene.set_camera_parameters(
            eye=np.array([278, 273, -770], dtype=np.float64),
            at=(np.array([278, 273, -769], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=int(39)
        )

        scene.add_geometries([
            Sphere(60, np.array([213 + 65, 450, 227 + 105 / 2 - 100]),
                   Le=1.25 * np.array([15.6, 15.6, 15.6])),
            Mesh("cbox_floor.npz",
                 brdf_params=np.array([0.86, 0.86, 0.86, 1])),
            Mesh("cbox_ceiling.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_back.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 50])),
            Mesh("cbox_greenwall.npz",
                 brdf_params=np.array([0.16, 0.76, 0.16, 1])),
            Mesh("cbox_redwall.npz",
                 brdf_params=np.array([0.76, 0.16, 0.16, 1])),
            Mesh("cbox_smallbox.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_largebox.npz",
                 brdf_params=np.array([0.86, 0.86, 0.86, 1000]))
        ])

        #########################################################################
        ### Deliverable 1: Implicit BRDF Sampling
        #########################################################################
        if enable_deliverables[0]:
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=2,
                                            sampling_type=IMPLICIT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=3,
                                            sampling_type=IMPLICIT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=4,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)

        #########################################################################
        ### Deliverable 2: ECSE 546 Only - Explicit Light BRDF Sampling
        #########################################################################
        if enable_deliverables[1]:
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=2,
                                            sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=3,
                                            sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=4,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)