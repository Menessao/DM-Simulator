import configparser
import numpy as np
import os

def read_config(TN):
    """
    Returns a list of parameters read from a configuration .yaml file

    Parameters
    ----------
    TN : string
        Name to the configuration file.

    Returns
    -------
    dm_par : list
        Deformable mirror parameters list.
    opt_par : list
        Optical parameters list.

    """
    # setup configparser
    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config_path = './ConfigFiles/' + TN + '.yaml'
    config.read(config_path)

     # DM configuration parameters
    dm_conf = config['DM']
    gap = dm_conf['hex_gap']
    l_hex = dm_conf['hex_side']
    n_rings = dm_conf['n_rings']
    act_pitch = dm_conf['act_pitch']
    act_r = dm_conf['act_radius']
    center = dm_conf['center_segment']

    dm_par = np.array([gap, l_hex, n_rings, act_pitch, act_r, center])
    dm_par = dm_par.astype(float)

    # Optical configuration parameters
    opt_conf = config['OPT']
    pix_scale = opt_conf['pixel_scale']
    pup_x = opt_conf['pupil_x']
    pup_y = opt_conf['pupil_y']
    opt_rad = opt_conf['opt_radius']

    opt_par = np.array([pix_scale, opt_rad, pup_x, pup_y])
    opt_par = opt_par.astype(float)

    
    # Mechanical parameters
    mech_conf = config['MECH']
    RoC = mech_conf['RoC']
    thk = mech_conf['thk']
    len = dm_conf['hex_side']
    E = mech_conf['E']
    rho = mech_conf['rho']
    nu = mech_conf['nu']

    mech_par = np.array([RoC,thk,len,E,rho,nu])
    mech_par = mech_par.astype(float)

    return dm_par, opt_par, mech_par
