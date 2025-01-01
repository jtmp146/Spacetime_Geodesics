import numpy as np
import sympy as sp
import time as t
import os

def flat_cart(x):
    """Generate Minkowski metric in Cartesian coordinates

    Args:
        x (tuple): Coordinates

    Returns:
        sympy.matrices.dense.MutableDenseMatrix: Cartesian Minkowski metric matrix
    """    
    ct, x1, x2, x3 = x
    g = sp.diag(-1, 1, 1, 1)
    return g

def flat_sph(x):
    """Generate Minkowski metric in spherical coordinates

    Args:
        x (tuple): Coordinates

    Returns:
        sympy.matrices.dense.MutableDenseMatrix: Spherical Minkowski metric matrix
    """    
    ct, x1, x2, x3 = x
    g = sp.diag(-1, 1, x1**2, (x1*sp.sin(x2))**2)
    return g

def schwarzschild(x, r_s):
    """Generate Schwarzschild metric in schwarzschild coordinates

    Args:
        x (tuple): Coordinates
        r_s (float): Schwarzschild radius

    Returns:
        sympy.matrices.dense.MutableDenseMatrix: Schwarzschild metric matrix in Schwarzschild coordinates
    """    
    ct, x1, x2, x3 = x
    g = sp.diag(-(1-r_s/x1), 1/(1-r_s/x1), x1**2, (x1*sp.sin(x2))**2)
    return g

def wave(x, h_plus, h_cross, w, c):
    """Generate metric for small-amplitude GWs in TT gauge

    Args:
        x (tuple): Coordinates
        h_plus (float): Plus-polarisation amplitude
        h_cross (float): Cross-polarisation amplitude
        w (float): Angular frequency
        c (float): Speed of light

    Returns:
        sympy.matrices.dense.MutableDenseMatrix: GW metric matrix in TT gauge
    """    
    ct, x1, x2, x3 = x
    gxx = 1+h_plus*sp.cos(w/c*(x3-ct))
    gyy = 1-h_plus*sp.cos(w/c*(x3-ct))
    gxy = h_cross*sp.cos(w/c*(x3-ct))
    g = sp.Matrix([[-1, 0, 0, 0], [0, gxx, gxy, 0], [0, gxy, gyy, 0], [0, 0, 0, 1]])
    return g

def schwarzschild_GP(x, r_s):
    """Generate Schwarzschild metric in Gullstrand-Painlevé coordinates

    Args:
        x (tuple): Coordinates
        r_s (float): Schwarzschild radius

    Returns:
        sympy.matrices.dense.MutableDenseMatrix: Schwarzschild metric matrix in GP coordinates
    """    
    ct, x1, x2, x3 = x
    a_primed = -1/(1-r_s/x1)*sp.sqrt(r_s/x1)
    g = sp.Matrix([[-(1-r_s/x1), -(1-r_s/x1), 0, 0], [-(1-r_s/x1), 1/(1-r_s/x1)-(1-r_s/x1)*a_primed**2, 0, 0], [0, 0, x1**2, 0], [0, 0, 0, (x1*sp.sin(x2))**2]])
    return g

def mag(v, g):
    """Calculate magnitude of vector in metric space

    Args:
        v (numpy.ndarray): Vector
        g (numpy.ndarray): Metric matrix

    Returns:
        float: Magnitude of v
    """    
    mag = 0
    for i in range(4):
        for j in range(4):
            mag += g[i,j]*v[i]*v[j]
    return mag**0.5

def gamma(i, j, k, x, g):
    """Generates Christoffel symbols of the second kind

    Args:
        i (int): Upper index
        j (int): 1st lower index
        k (int): 2nd lower index
        x (tuple): Coordinates
        g (sympy.matrices.dense.MutableDenseMatrix): Metric matrix

    Returns:
        sympy equation: Christoffel symbol with indices i, j, k
    """    
    gamma = 0
    for l in range(4):
        gamma += 0.5*g.inv()[i, l]*(sp.diff(g[l,k], x[j])+sp.diff(g[l,j], x[k])-sp.diff(g[j,k], x[l]))
    return gamma

def four_acc_i(i, x, u_sym, g):
    """Calculate four-vector acceleration at a point

    Args:
        i (int): Component of four-acceleration to calculate
        x (tuple): Coordinates
        u_sym (tuple): Four-velocity components
        g (sympy.matrices.dense.MutableDenseMatrix): Metric matrix

    Returns:
        sympy equation: ith component of four-acceleration
    """    
    four_acc = 0
    for j in range(4):
        for k in range(4):
            four_acc += -gamma(i, j, k, x, g)*u_sym[j]*u_sym[k]
    return four_acc

def trajectory(worldline):
    """Convert a wordline into an array of points equally spaced in time

    Args:
        worldline (numpy.ndarray): Array of 4D four-position coordinates

    Returns:
        numpy.ndarray: Array of 3D spatial position coordinates
    """    
    dt = np.min(np.diff(worldline[:, 0]))
    start = worldline[0, 0]
    end = worldline[-1, 0]
    divs = int((end-start)/dt)+1
    trajectory = np.empty(shape=(divs, 3))
    for i in range(divs):
        t = start+i*dt
        pos = worldline[np.argmin(np.abs(worldline[:, 0]-t)), 1:]
        trajectory[i] = pos
    return trajectory

def sph_to_cart(trajectory):
    """Converts trajectory of spherical coordinates to Cartesian coordinates

    Args:
        trajectory (numpy.ndarray): Trajectory in spherical coordinates

    Returns:
        numpy.ndarray: Trajectory in Cartesian coordinates
    """    
    cart_traject = np.empty(shape=trajectory.shape)
    for i in range(trajectory.shape[0]):
        r, theta, phi = trajectory[i]
        cart_traject[i] = r*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return cart_traject

def csv(trajectory, size):
    """Saves particle trajectory as a CSV file

    Args:
        trajectory (numpy.ndarray): Trajectory in Cartesian coordinates
        size (float): Particle size (for plotting)
    """    
    if os.path.exists("trajectory.txt"):
        os.remove("trajectory.txt")
    f = open("trajectory.txt", "a")
    for tau in range(trajectory.shape[0]):
        line = " ".join([str(x) for x in trajectory[tau]]).replace("[", "").replace("]", "")+f" {size}\n"
        f.write(line)
    f.close()

# def close_angle(theta):
#     """Ensures an angle theta is within 2pi and 0

#     Args:
#         theta (float): Angle

#     Returns:
#         float: Angle 
#     """    
#     lim = 2*np.pi
#     if theta > lim:
#         return theta%lim
#     elif theta < 0:
#         return lim-(-theta)%lim
#     else:
#         return theta

def init_cart(x, u_sym, start_range, vel_range, worldline, args):
    """Initialise simulation for Minkowski metric in cartesian coordinates

    Args:
        x (tuple): Coordinates
        u_sym (tuple): Four-velocity components
        start_range (float): Range of inital positions
        vel_range (float): Range of initial velocities
        worldline (numpy.ndarray): Array of 4D four-position vectors
        args (tuple): Needed to give all init methods same signiture

    Returns:
        tuple: metric matrix, initial four-position, whether or not spherical coordinates are used
    """    
    g = flat_cart(x)
    sph = False
    return g, worldline[0], sph

def init_sph(x, u_sym, start_range, vel_range, worldline, args):
    """Initialise simulation for Minkowski metric in spherical coordinates

    Args:
        x (tuple): Coordinates
        u_sym (tuple): Four-velocity components
        start_range (float): Range of inital positions
        vel_range (float): Range of initial velocities
        worldline (numpy.ndarray): Array of 4D four-position vectors
        args (tuple): Needed to give all init methods same signiture

    Returns:
        tuple: metric matrix, initial four-position, whether or not spherical coordinates are used
    """    
    g = flat_sph(x)
    sph = True
    worldline[0,1] = np.abs(worldline[0,1])
    worldline[0,2], worldline[0,3] = np.random.uniform(0, 2*np.pi, size=2)
    # worldline[0,2] = close_angle(worldline[0,2])
    # worldline[0,3] = close_angle(worldline[0,3])
    return g, worldline[0], sph

def init_schwarzschild(x, u_sym, start_range, vel_range, worldline, args):
    """Initialise simulation for Schwarzschild metric in desired coordinates

    Args:
        x (tuple): Coordinates
        u_sym (tuple): Four-velocity components
        start_range (float): Range of inital positions
        vel_range (float): Range of initial velocities
        worldline (numpy.ndarray): Array of 4D four-position vectors
        args (tuple): Schwarzschild radius and generation function for metric in desired coordinates

    Returns:
        tuple: metric matrix, initial four-position, whether or not spherical coordinates are used
    """    
    r_s, schwarzschild_coords = args
    g = schwarzschild_coords(x, r_s)
    sph = True
    worldline[0,1] = np.abs(np.random.uniform(r_s, r_s+start_range, size=1)[0])
    worldline[0,2], worldline[0,3] = np.random.uniform(0, 2*np.pi, size=2)
    # worldline[0,2] = close_angle(worldline[0,2])
    # worldline[0,3] = close_angle(worldline[0,3])
    return g, worldline[0], sph

def init_wave(x, u_sym, start_range, vel_range, worldline, args):
    """Initialise simulation for gravitational wave metric in TT gauge

    Args:
        x (tuple): Coordinates
        u_sym (tuple): Four-velocity components
        start_range (float): Range of inital positions
        vel_range (float): Range of initial velocities
        worldline (numpy.ndarray): Array of 4D four-position vectors
        args (tuple): Plus-polarisation amplitude, cross-polarisation amplitude, angular frequency and speed of light

    Returns:
        tuple: metric matrix, initial four-position, whether or not spherical coordinates are used
    """    
    h_plus, h_cross, w, c = args
    g = wave(x, h_plus, h_cross, w, c)
    sph = False
    return g, worldline[0], sph

def four_acc(x, u_sym, g):
    """Calculate four-acceleration vector

    Args:
        x (tuple): Coordinates
        u_sym (tuple): Four-velocity components
        g (sympy.matrices.dense.MutableDenseMatrix): Metric matrix

    Returns:
        sympy.matrices.dense.MutableDenseMatrix: Column matrix representing four-acceleration vector
    """    
    dudtau = sp.Matrix([0, 0, 0, 0])
    for i in range(4):
        dudtau[i] = four_acc_i(i, x, u_sym, g)
    return dudtau

def norm_v(v, g):
    """Normalise a vector in a metric space

    Args:
        v (numpy.ndarray): Vector
        g (numpy.ndarray): Metric matrix

    Returns:
        numpy.ndarray: Normalised vector
    """    
    return v/mag(v, g)

def init_conds(steps, x, u_sym, start_range, vel_range, args, init_metric):
    """_summary_

    Args:
        steps (int): Number of simulation steps to be performed
        x (tuple): Coordinates
        u_sym (tuple): Four-velocity components
        start_range (float): Range of inital positions
        vel_range (float): Range of initial velocities
        worldline (numpy.ndarray): Array of 4D four-position vectors
        args (tuple): Additional arguments specific to desired metric
        init_metric (function): Metric initialisation function to use

    Returns:
        tuple: worldline array, four-velocity array, metric matrix, whether or not spherical coordinates are used
    """    
    worldline = np.empty(shape=(steps, 4))
    worldline[0] = np.random.uniform(-start_range, start_range, size=4)
    worldline[0,0] = 0
    g, worldline[0], sph = init_metric(x, u_sym, start_range, vel_range, worldline, args)
    four_vel = np.empty(shape=(steps, 4))
    four_vel[0] = np.random.uniform(-vel_range, vel_range, size=4)
    four_vel[0,0] = np.abs(four_vel[0,0])
    four_vel[0] = c*norm_v(four_vel[0], g.evalf(subs={ct:worldline[0,0], x1:worldline[0,1], x2:worldline[0,2], x3:worldline[0,3]}))
    print("\nStart 4-position:", worldline[0])
    print("Start 4-velocity:", four_vel[0])
    print("4-velocity magnitude:", mag(four_vel[0], g.evalf(subs={ct:worldline[0,0], x1:worldline[0,1], x2:worldline[0,2], x3:worldline[0,3]})))
    return worldline, four_vel, g, sph

def simulate(steps, worldline, four_vel, dudtau, dtau):
    """Use Euler method to generate geodesic worldline given inital conditions

    Args:
        steps (int): Number of proper timesteps to simulate
        worldline (numpy.ndarray): Array of four-position vectors
        four_vel (numpy.ndarray): Array of four-velocity vectors
        dudtau (sympy equation): Four-acceleration equation
        dtau (float): Proper time between steps

    Returns:
        numpy.ndarray: Array of simulated four-position vectors
    """    
    for j in range(1, steps):
        q = tuple(worldline[j-1])
        u = tuple(four_vel[j-1])
        for i in range(4):
            four_vel[j, i] = u[i]+dtau*dudtau[i].evalf(subs={ct:q[0], x1:q[1], x2:q[2], x3:q[3], u0:u[0], u1:u[1], u2:u[2], u3:u[3]})
            worldline[j,i] = q[i]+dtau*u[i]
    return worldline

def remove_sing(worldline):
    """Removes parts of worldline if a singularity is encountered

    Args:
        worldline (numpy.ndarray): Array of four-position vectors

    Returns:
        numpy.ndarray: Worldline with any singularities removed
    """    
    nans = (worldline == np.nan).astype(int)
    infs = (worldline == np.inf).astype(int)
    sings = nans + infs
    sings_i = np.where(sings > 0)[0]
    if sings_i.shape[0] == 0:
        print("No singularities found.")
        return worldline
    else:
        print("Worldline contains singularity.")
        sing_i = np.min(sings_i)
        return worldline[0:sing_i]

start = t.time()
steps = 1000
start_range = 2
vel_range = 0.5
dtau = 0.01
size = 50

ct, x1, x2, x3 = sp.symbols("ct, x1, x2, x3")
u0, u1, u2, u3 = sp.symbols("u0, u1, u2, u3")
u_sym = (u0, u1, u2, u3)
x = (ct, x1, x2, x3)
c = 1

# For testing:
# np.random.seed(128)

worldline, four_vel, g, sph = init_conds(steps, x, u_sym, start_range, vel_range, (2, schwarzschild), init_schwarzschild)
print("\nCalculating four acceleration")
dudtau = four_acc(x, u_sym, g)
print("\nStarting simulation")
worldline = simulate(steps, worldline, four_vel, dudtau, dtau)
worldline = remove_sing(worldline)
trajectory = trajectory(worldline)

if sph:
    print("\nChanging to Cartesian coordinates")
    trajectory = sph_to_cart(trajectory)

print("\nFinal radial distance:", np.sum(trajectory[-1]**2)**0.5)
print("Final 4-velocity:", four_vel[-1])
csv(trajectory, size)

end = t.time()
print(f"\nRuntime: {end-start}")