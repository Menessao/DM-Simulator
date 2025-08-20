import numpy as np
import matplotlib.pyplot as plt

from segmented_deformable_mirror import SegmentedMirror
from segment_geometry import HexagonGeometry
from matrix_calculator import matmul


def define_dsm(TN:str, global_zern_radial_order:int = 6, local_zern_radial_order:int = 5):
    """
    Performs the basic operations to define a segmented
    mirror object from the data in the configuration file

    Parameters
    ----------
    TN : string
        Configuration file tracking number.
        
    global_zern_radial_order : int, optional
        Radial order of Zernike modes for the IM on the global mask. 
        The default is 6.
        
    local_zern_radial_order : int, optional
        Radial order of Zernike modes for the IM on the single segment. 
        The default is 5.

    Returns
    -------
    sdm : segmented deformable mirror class
        Segmented deformable mirror object.

    """
    print('Defining mirror geometry ...')
    geom = HexagonGeometry(TN)
    
    print('Initializing segmented mirror class ...')
    sdm = SegmentedMirror(geom)

    n_global_zern = global_zern_radial_order*(global_zern_radial_order-1)//2
    n_local_zern = local_zern_radial_order*(local_zern_radial_order-1)//2
    
    # Global Zernike matrix
    print('Computing ' + str(n_global_zern) + ' modes global Zernike interaction matrix ...')
    sdm.compute_global_zern_matrix(n_global_zern)
    tiptiltfocus = np.zeros(n_global_zern)
    tiptiltfocus[1] = 1
    tiptiltfocus[2] = 1
    tiptiltfocus[3] = 1
    wf = matmul(sdm.glob_ZM,tiptiltfocus)
    sdm.acquire_map(wf, plt_title='Global Tip/Tilt + Focus')
    
    # Local Zernike matrix
    print('Computing ' + str(n_local_zern) + ' modes local Zernike interaction matrix ...')
    sdm.compute_local_zern_matrix(n_local_zern)
    INTMAT = sdm.ZM
    n_hex = np.shape(INTMAT)[0]
    cmd_ids = np.arange(n_local_zern-1)+1
    cmd_ids = np.tile(cmd_ids,int(np.ceil(n_hex/(n_local_zern-1))))
    cmd_ids = cmd_ids[0:n_hex]
    cmd_ids = cmd_ids + n_local_zern*np.arange(n_hex)
    modal_cmd = np.zeros(n_hex*n_local_zern)
    modal_cmd[cmd_ids] = 1
    flat_shape = matmul(INTMAT,modal_cmd)
    sdm.acquire_map(flat_shape, plt_title='Zernike modes')
    
    # Global influence functions and global reconstructor
    print('Initializing influence functions and reconstructor matrices ...')
    sdm.initialize_IFF_and_R_matrices()
    cmd_for_zern = matmul(sdm.R,flat_shape)
    flat_img = matmul(sdm.IFF,cmd_for_zern)
    sdm.acquire_map(flat_img, plt_title='Reconstructed Zernike modes')
    
    n_acts = np.shape(sdm.IFF)[2]
    cmd = np.zeros(n_hex*n_acts)
    cmd_ids = np.arange(n_acts)
    cmd_ids = np.tile(cmd_ids,int(np.ceil(n_hex/(n_acts))))
    cmd_ids = cmd_ids[0:n_hex]
    cmd_ids = cmd_ids + n_acts*np.arange(n_hex)
    cmd[cmd_ids] = 1 #np.ones(n_hex)
    flat_img = matmul(sdm.IFF,cmd)
    sdm.acquire_map(flat_img, plt_title='Actuators influence functions')
    
    return sdm


def fitting_error_plots(mask, IM, IFF, R, pix_ids = None):
    """
    Computes the fitting error for the reconstructor on a mask,
    plotting the N = np.shape(IM)[1] residual images

    Parameters
    ----------
    mask : bool ndarray [Npix,]
        Boolean mask of the flattened image.
    IM : float ndarray [Npix,Nmodes]
        Interaction matrix of the images to fit.
    IFF : float ndarray [Npix,Nacts]
        Influence functions matrix from actuators data.
    R : float ndarray [Nacts,Npix]
        Reconstructor matrix (IFF pseudo-inverse).
    pix_ids : ndarray(int), optional
        The indices of the valid pixel indices on the mask. The default is None.

    Returns
    -------
    RMS_vec : float ndarray [Nmodes,]
        Vector of ratios of the STDs of the reconstructed
        image and the desired Zernike mode from the IM.

    """
    
    N = np.shape(IM)[-1]
    RMS_vec = np.zeros(N)
    
    flat_img = np.zeros(np.size(mask))
    flat_mask = mask.flatten()
    cube = np.zeros([N,np.shape(mask)[0],np.shape(mask)[1]])
    
    for k in range(N):
        des_shape = IM[:,k]
        act_cmd = matmul(R,des_shape)
        act_shape = matmul(IFF,act_cmd)
        shape_err = des_shape-act_shape
            
        shape_RMS = np.std(des_shape)
        if shape_RMS < 1e-15: # avoid division by zero
            shape_RMS = 1
        RMS_vec[k] = np.std(shape_err)/shape_RMS
        
        if pix_ids is None:
            pix_ids = ~flat_mask
            
        flat_img[pix_ids] = shape_err
        img = np.reshape(flat_img, np.shape(mask))
        cube[k] = img

    n_cols = 5
    n_rows = int(np.ceil(N/n_cols))
    plt.figure(figsize=(3.3*n_cols,3.2*n_rows))
    for i in range(N):
        plt.subplot(n_rows,n_cols,i+1)
        plt.imshow(np.ma.masked_array(cube[i], mask), origin = 'lower', cmap = 'inferno')
        plt.colorbar()
        plt.title(f"Mode {i+1} shape error\n RMS: {RMS_vec[i]:.2e}")
        
    plt.figure()
    plt.plot(np.arange(N)+1, RMS_vec,'-o',color='orange')
    plt.xlabel('Noll id')
    plt.ylabel('RMS')
    plt.title('Zernike fitting error')
    plt.xticks(np.arange(N)+1)
    plt.grid('on')
    # plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])
    plt.xlim([0.5,N+0.5])
    
    return RMS_vec


def fitting_error(IM, IFF, R):
    """
    Computes the fitting error for the reconstructor

    Parameters
    ----------
    IM : float ndarray [Npix,Nmodes]
        Interaction matrix of the images to fit.
    IFF : float ndarray [Npix,Nacts]
        Influence functions matrix from actuators data.
    R : float ndarray [Nacts,Npix]
        Reconstructor matrix (IFF pseudo-inverse).

    Returns
    -------
    RMS_vec : float ndarray [Nmodes,]
        Vector of ratios of the STDs of the reconstructed
        image and the desired image from the IM.

    """
    
    act_cmds = matmul(R,IM)
    act_shapes = matmul(IFF,act_cmds)
    res_shapes = IM - act_shapes
    
    img_rms = np.std(IM,axis = 0)
    img_rms[img_rms < 1e-15] = np.ones(len(img_rms[img_rms < 1e-15])) # avoid division by 0
    RMS_vec = np.std(res_shapes,axis = 0)/img_rms
    
    return RMS_vec
        
        
def update_act_coords_on_ring(dm, n_ring:int, do_save:bool = False):
    """
    Update the coordinates of all segments on a given ring on a 
    segmented deformable mirror

    Parameters
    ----------
    dm : segmented deformable mirror class
        The segmented deformable mirror object.
    n_ring : int
        The integer ring number, the segments on which will have their coordinates updated.

    Returns
    -------
    None.

    """
    print("Updating actuator coordinates on ring, hang tight...")
    n_hexagons = lambda n: int(1 + (6 + n*6)*n/2)
    
    hex_ids = np.array([0])
    
    # Get hex ring ids
    if n_ring > 0:
        hex_ids = np.arange(n_hexagons(n_ring-1),n_hexagons(n_ring))-1+dm.geom.center_bool
    
    # Define new coordinates from the old ones
    old_coords = dm.segment[0].act_coords - dm.segment[0].center
    d = np.sqrt(old_coords[:,0]**2+old_coords[:,1]**2)
    thr = 0.8
    new_coords = old_coords.copy()
    new_coords[d>thr] = old_coords[d>thr]*0.9
    
    # Plot to check new coordinates
    c_hex = dm.geom.hex_outline
    plt.figure()
    plt.plot(c_hex[0],c_hex[1]) 
    plt.scatter(new_coords[:,0],new_coords[:,1])
    
    # Update coordinates accordingly
    dm.update_act_coords(hex_ids, new_coords, do_save)


    
    
# def capsens_measure(dm, segment_id):
#     """ [WIP] """
    
#     act_rad = 0.02 #dm.geom.act_radius
#     capsens_rad = 0.04 #dm.geom.act_pitch/2.2
    
#     segm = dm.segment[segment_id]
    
#     CSMAT = define_capsens_matrix(segm.mask, dm.geom.pix_scale, segm.act_coords, act_rad, capsens_rad)
#     capsens_flat_img = np.sum(CSMAT,axis=0)
#     segm.surface(capsens_flat_img)
#     segm.get_position(act_rad*dm.geom.pix_scale)
#     meas_gap = CSMAT @ segm.shape
    
#     return meas_gap
    



