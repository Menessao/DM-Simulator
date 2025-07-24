import numpy as np
from tps import ThinPlateSpline # for the simulated IFF

#import os
#import subprocess
from scipy.interpolate import griddata

from zernike_polynomials import generate_zernike_matrix as assemble_zern_mat


def matmul(mat, vec):
    """
    Simple function to perform matrix multiplication
    for both block matrices (saved as an array of
    matrices) and regular matrices
    """
    
    if len(np.shape(mat)) == 3: # Array of matrices
        id_ctr = 0 
        n_hex, n_pix, N = np.shape(mat)
        res = np.zeros(n_pix*n_hex)
        for n, mat_n in enumerate(mat):
            ids = np.arange(id_ctr, id_ctr+N)
            res[n_pix*n:n_pix*(n+1)] = mat_n @ vec[ids]
            id_ctr += N
    else:
        res = mat @ vec
        
    return res

def compute_reconstructor(M, thr:float = 0.):
    """
    Computes the reconstructor (pseudo-inverse) 
    for the interaction matrix M.

    Parameters
    ----------
    M : ndarray(float) [Npix,N]
        Interaction matrix to be inverted.
        
    thr : float, optional
        Threshold for the inverse eigenvalues. 
        The eigenvalues v  s. t. 1/v < thr 
        are discarded when computing the inverse.
        By default, no eigenvalues are discarded.

    Returns
    -------
    Rec : ndarray(float) [N,Npix]
        The pseudo-inverse of M.

    """
    
    U,S,V = np.linalg.svd(M, full_matrices=False)
    Sinv = 1/S
    Sinv[Sinv < thr] = 0
    Rec = (V.T * Sinv) @ U.T
    return Rec


# def compute_mirror_modes(K):
#     """ Computes the mirror modes from the stiffness matrix,
#     returning the eigenmodes and eignevalues sorted in order
#     of increasing stiffness """

#     _,lambdas,V = np.linalg.svd(K, full_matrices=False)

#     eigvals = np.sort(lambdas)
#     col_ids = np.argsort(lambdas)

#     eigmodes = V.T[:,col_ids]

#     return eigmodes, eigvals

# def simulate_influence_functions(act_coords, local_mask, pix_scale, amps = 1.0):
#     """ Simulate the influence functions by 
#     imposing 'perfect' zonal commands """
    
#     n_acts = len(act_coords)
#     H,W = np.shape(local_mask)
    
#     pix_coords = get_pixel_coords(local_mask, pix_scale)
#     act_pix_coords = get_pixel_coords(local_mask, pix_scale, act_coords)
    
#     img_cube = np.zeros([H,W,n_acts])
    
#     if isinstance(amps,float):
#         amps *= np.ones(n_acts)

#     for k in range(n_acts):
#         act_data = np.zeros(n_acts)
#         act_data[k] = amps[k]
#         tps = ThinPlateSpline(alpha=0.0)
#         tps.fit(act_pix_coords, act_data)
#         flat_img = tps.transform(pix_coords)
#         uint8_img = scale2uint8(flat_img) # scale to uint8
#         img_cube[:,:,k] = np.reshape(uint8_img, [H,W])

#     # Masked array
#     cube_mask = np.tile(local_mask,n_acts)
#     cube_mask = np.reshape(cube_mask, np.shape(img_cube), order = 'F')
#     cube = np.ma.masked_array(img_cube, cube_mask, dtype=np.uint8)
    
#     return cube

# def get_pixel_coords(mask, pix_scale:float, coords = None):
#     """ Convert x,y coordinates in coords to pixel coordinates
#     or get the pixel coordinates of a mask

#     Parameters
#     ----------
#     mask : ndarray(bool)
#         The image mask where the pixels are.
        
#     pix_scale : float
#         The number of pixels per meter.
        
#     coords : ndarray(float) [N,2], optional
#         The N coordinates to convert in pixel coordinates.
#         Defaults to all pixels on the mask.

#     Returns
#     -------
#     pix_coords : ndarray(int)
#         The obtained pixel coordinates.

#     """
    
#     H,W = np.shape(mask)
    
#     if coords is not None:
#         pix_coords = np.zeros([len(coords),2],dtype=int)
#         pix_coords[:,0] = np.round(coords[:,1]*pix_scale + H/2)
#         pix_coords[:,1] = np.round(coords[:,0]*pix_scale + W/2)
#     else:
#         pix_coords = np.zeros([H*W,2])
#         pix_coords[:,0] = np.repeat(np.arange(H),W)
#         pix_coords[:,1] = np.tile(np.arange(W),H)
    
#     return pix_coords


def slaving(coords, cmd, slaving_method:str = 'interp', cmd_thr:float = None):
    """ 
    Clip the command to avoid saturation

    Parameters
    ----------
    coords : ndarray(float) [2,Nacts]
        The actuator coordinates

    cmd : ndarray(float) [Nacts]
        The mirror command to be slaved

    cmd_thr : float, optional
        Threshold to define slave actuators.
        Slave acts are defined as the actuators for which: cmd > cmd_thr.
        Default: slave acts the ones outside 3 sigma of the mean cmd
        
    slaving_method : str, optional
        The way to treat slave actuators.
            'tps'     : thin plate spline interpolation. DEFAULT
            'zero'    : set slave actuator command to zero 
            'clip'    : clips the actuator commands to the given cmd_thr input
            'nearest' : nearest actuators interpolation
            'wmean'   : mean of nearby actuators weighted on 1/r^2

            'exclude' : exclude slaves from the reconstructor computation (TBI)

    """
    # Define master ids
    n_acts = len(cmd)
    act_ids = np.arange(n_acts)

    if cmd_thr is None:
        master_ids = act_ids[np.abs(cmd-np.mean(cmd)) <= 3*np.std(cmd)]
    else:
        master_ids = act_ids[np.abs(cmd) <= cmd_thr]
    
    match slaving_method:

        case 'tps':
            tps = ThinPlateSpline(alpha=0.0)
            tps.fit(coords[master_ids], cmd[master_ids])
            rescaled_cmd = tps.transform(coords)
            slaved_cmd = rescaled_cmd[:,0]

        case 'zero':
            pad_cmd = np.zeros_like(cmd)
            pad_cmd[master_ids] = cmd[master_ids]
            slaved_cmd = pad_cmd

        case 'clip':
            slaved_cmd = np.minimum(np.abs(cmd), cmd_thr)
            slaved_cmd *= np.sign(cmd)

        case 'nearest':
            master_coords = coords[:,master_ids]
            slaved_cmd = griddata(master_coords, cmd[master_ids], (coords[0], coords[1]), method='nearest')

        case 'wmean':
            master_coords = coords[:,master_ids]
            master_cmd = cmd[master_ids]
            dist2 = lambda xy: (xy[0]-master_coords[0])**2 + (xy[1]-master_coords[1])**2 
            is_slave = np.ones_like(cmd, dtype=bool)
            is_slave[master_ids] = False
            slave_ids = act_ids[is_slave]
            slaved_cmd = cmd.copy()
            for slave in slave_ids:
                d2_slave = dist2(coords[:,slave])
                weighted_cmd = master_cmd / d2_slave
                slaved_cmd[slave] = np.sum(weighted_cmd)*np.sum(d2_slave)/n_acts

        # case 'exclude':
        #     masked_IFF = self.IFF[valid_ids,:]
        #     masked_IFF = masked_IFF[:,visible_acts]
        #     masked_R = np.linalg.pinv(masked_IFF)
            
        #     pad_cmd = np.zeros_like(act_cmd)
        #     pad_cmd[visible_acts] = matmul(masked_R, masked_shape)
        #     act_cmd = pad_cmd
            
        case _:
            raise NotImplementedError(f"{slaving_method} is not an available slaving method. Available methods are: 'tps', 'zero', 'clip', 'nearest', 'wmean'")

    return slaved_cmd

def interpolate_influence_functions(iffs, in_mesh, npix = np.array([256,256],dtype=int) ):
    """ Interpolates the influence functions defined on the in_mesh
    to a new grid of npix by npix points """

    npix_x, npix_y = npix

    x, y = in_mesh[:,0], in_mesh[:,1]
    new_x = np.linspace(min(x), max(x), npix_x)
    new_y = np.linspace(min(y), max(y), npix_y)
    gx, gy = np.meshgrid(new_x, new_y)

    interp_iffs = griddata((x, y), iffs, (gx, gy), method='linear')

    return interp_iffs


def cube2mat(cube):
    """ Get influence functions matrix 
    from the image cube """
    
    n_acts = np.shape(cube)[2]
    valid_len = int(np.sum(1-cube.mask)/n_acts)
    
    flat_cube = cube.data[~cube.mask]
    local_IFF = np.reshape(flat_cube, [valid_len, n_acts])
    
    IFF = np.array(local_IFF)
    
    return IFF
    


def compute_zernike_matrix(mask, n_modes):
    """ Computes the zernike matrix: 
        [n_pixels,n_modes] """
    
    noll_ids = np.arange(n_modes) + 1
    mat = assemble_zern_mat(noll_ids, mask)
    
    return mat   


def scale2uint8(img):
    """ Scales an ndarray to uint8 """
    
    img -= np.min(img)
    img = img/np.max(img) * (2**8-1)
    img = np.round(img)
    
    return (img).astype(np.uint8)


def simulate_influence_functions(act_coords, local_mask, pix_scale:float = 1.0):
    """ Simulate the influence functions by 
    imposing 'perfect' zonal commands """
    
    n_acts = np.max(np.shape(act_coords))
    mask_ids = np.arange(np.size(local_mask))
    pix_ids = mask_ids[~(local_mask).flatten()]
    
    pix_coords = getMaskPixelCoords(local_mask).T

    # H,W = np.shape(local_mask)
    # act_pix_coords = np.zeros_like(act_coords)
    # act_pix_coords[0] = (act_coords[1]*pix_scale + H/2)
    # act_pix_coords[1] = (act_coords[0]*pix_scale + W/2)
    # act_pix_coords = np.round(act_pix_coords).T
    act_pix_coords = get_pixel_coords(local_mask, act_coords, pix_scale).T

    IFF = np.zeros([len(pix_ids),n_acts])

    for k in range(n_acts):
        act_data = np.zeros(n_acts)
        act_data[k] = 1e-6
        tps = ThinPlateSpline(alpha=0.0)
        tps.fit(act_pix_coords, act_data)
        img = tps.transform(pix_coords)
        IFF[:,k] = img[pix_ids,0]

    return IFF


def get_pixel_coords(mask, coords, pix_scale):
    """ 
    Convert x,y coordinates in coords to pixel coordinates
    or get the pixel coordinates of a mask

    Parameters
    ----------
    mask : ndarray(bool)
        The image mask where the pixels are.
        
    coords : ndarray(float) [2,N]
        The N coordinates to convert in pixel coordinates.
        Defaults to all pixels on the mask.

    pix_scale : float (Optional)
        The number of pixels per meter.
        

    Returns
    -------
    pix_coords : ndarray(int) [2,N]
        The obtained pixel coordinates.
    """
    
    H,W = np.shape(mask)

    pix_coords = np.zeros_like(coords)#([2,np.shape(coords)[1]])
    pix_coords[0] = (coords[1]*pix_scale + H/2)
    pix_coords[1] = (coords[0]*pix_scale + W/2)
    # pix_coords[0] = (coords[1]*pix_scale/2 + H)/2
    # pix_coords[1] = (coords[0]*pix_scale/2 + W)/2
    # pix_coords[0,:] = (coords[1,:]*pix_scale/2 + H)/2
    # pix_coords[1,:] = (coords[0,:]*pix_scale/2 + W)/2
    
    return pix_coords


def get_coords_from_IFF(IFF, mask, use_peak=True):
    """
    Get the coordinates of the actuators from the influence functions matrix.

    Parameters
    ----------
    IFF : ndarray(float) [Npix,Nacts]
        The influence functions matrix.
        
    mask : ndarray(bool) [Npix,Npix]
        The DM mask.
    
    use_peak : bool, optional
        If True, actuator coordinates are computed from the IFF peak.
        If False, actuator coordinates are computed from the photocenter of the IFF.
        Defaults to True, the photocenter approach seems to be giving issues

    Returns
    -------
    coords : ndarray(float) [2,Nacts]
        The coordinates of the actuators in the mask.
    """
    
    # Get pixel coordinates
    pix_coords = getMaskPixelCoords(mask)

    x_coords = pix_coords[0,:]
    y_coords = pix_coords[1,:]

    mask = (mask).astype(bool) # ensure mask is boolean
    x_coords = x_coords[~mask.flatten()]
    y_coords = y_coords[~mask.flatten()]

    # Get the coordinates of the actuators
    n_acts = IFF.shape[1]
    act_coords = np.zeros([2, n_acts])

    for k in range(n_acts):
        act_data = IFF[:, k]
        if use_peak:
            max_id = np.argmax(act_data)
            act_coords[0,k] = x_coords[max_id]
            act_coords[1,k] = y_coords[max_id] 
        else:
            act_coords[0,k] = np.sum(x_coords * act_data) / np.sum(act_data)
            act_coords[1,k] = np.sum(y_coords * act_data) / np.sum(act_data)

    return act_coords


def getMaskPixelCoords(mask):
    """ 
    Get the pixel coordinates of a mask

    Parameters
    ----------
    mask : ndarray(bool)
        The image mask where the pixels are.

    Returns
    -------
    pix_coords : ndarray(int) [2,N]
        The obtained pixel coordinates.
    """
    
    H,W = np.shape(mask)
    pix_coords = np.zeros([2,H*W])
    pix_coords[0,:] = np.repeat(np.arange(H),W)
    pix_coords[1,:] = np.tile(np.arange(W),H)
    
    return pix_coords


def define_capsens_matrix(mask, pix_scale, act_coords, r_in, r_out, capsens_coords = None):
    """
    Determine the pixel CapSens matrix to estimate the gap measured by
    the capacitive sensors for a give shape as meas_gap = CSMAT @ shape

    Parameters
    ----------
    mask : ndarray(bool)
        The mask representing the mirror on which to operate.
    pix_scale : float
        The number of pixels per meter.
    act_coords : ndarray(float) [Nacts,2]
        The local actuator coordinates (in meters).
    r_in : float
        The CapSens inner radius (in meters).
    r_out : float
        The CapSens outer radius (in meters).
    capsens_coords : ndarray(float) [Nacts,2], optional
        The local actuator coordinates (in meters).
        Useful if there is an eccentricity between 
        CapSens and actuators. The default is act_coords.

    Returns
    -------
    CSMAT : ndarray(float) [Nacts,Npixels]
        The obtained CapSens pixel matrix, normalized by the
        pixel area (i.e. number of pixels per CapSens)

    """
    
    if capsens_coords is None:
        capsens_coords = act_coords
        
    X,Y = np.shape(mask)
    d = lambda i,j: np.sqrt(i**2+j**2)

    pix_act_coords = get_pixel_coords(mask, pix_scale, act_coords)
    pix_capsens_coords = get_pixel_coords(mask, pix_scale, capsens_coords)
    
    pix_in = int(r_in*pix_scale)
    pix_out = int(r_out*pix_scale)
    
    CSMAT = np.zeros([len(act_coords),np.sum(1-mask)])
    
    for k, pix_act_coord in enumerate(pix_act_coords):
        act_x, act_y = pix_act_coord
        sens_x, sens_y = pix_capsens_coords[k,:]
        sensor = np.fromfunction(lambda i,j: (d(i-act_y,j-act_x) >= pix_in) * (d(i-sens_y,j-sens_x) < pix_out), [X,Y])
        # img = np.ma.masked_array(sensor,mask), plt.figure(), plt.imshow(img, origin = 'lower')
        
        masked_data = sensor[~mask]
        pix_area = np.sum(masked_data)
        CSMAT[k,:] = masked_data/pix_area
    
    return CSMAT

    




    