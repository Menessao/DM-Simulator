import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# from segment_geometry import circular_mask
import utilities as utils

tn = '20240920'
dsm = utils.define_dsm(tn)

# # Fitting errors
# seg0 = dsm.segment[0]
# local_fit_err = utils.fitting_error_plots(seg0.mask, seg0.ZM, seg0.IFF, seg0.R)

# global_fit_err = utils.fitting_error_plots(dsm.mask, dsm.glob_ZM, dsm.IFF, dsm.R, dm.valid_ids.flatten())

# # Update coordinates
# utils.update_act_coords_on_ring(dsm, 0)
# utils.update_act_coords_on_ring(dsm, 2)

# # Define anular mask
# r_in = 1.
# r_out = 5.
# mask_in = circular_mask(r_in,dsm.geom.pix_scale,np.shape(dsm.mask))
# mask_out = circular_mask(r_out,dsm.geom.pix_scale,np.shape(dsm.mask))
# anular_mask = np.logical_or(1-mask_in,mask_out)

# # Segment Scramble
# dsm.segment_scramble(reset_shape=True)
# dsm.acquire_map(plt_mask = anular_mask)

# dsm.apply_masked_flat(mask = anular_mask)
# dsm.get_position()
# dsm.acquire_map(plt_mask = anular_mask)
# dsm.acquire_map()

# dsm.segment_scramble(reset_shape=True)
# dsm.apply_masked_flat(mask = anular_mask, slaving_mode='zero')
# dsm.get_position()
# dsm.acquire_map(plt_mask = anular_mask)
# dsm.acquire_map()

# dsm.segment_scramble(reset_shape=True)
# dsm.apply_masked_flat(mask = anular_mask, slaving_mode='interp')
# dsm.get_position()
# dsm.acquire_map(plt_mask = anular_mask)
# dsm.acquire_map()

# dsm.segment_scramble(reset_shape=True)
# dsm.apply_masked_flat(mask = anular_mask, slaving_mode='exclude')
# dsm.get_position()
# dsm.acquire_map(plt_mask = anular_mask)
# dsm.acquire_map()
