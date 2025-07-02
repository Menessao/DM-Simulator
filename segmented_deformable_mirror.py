import numpy as np
from mirror_segment import Segment
import matrix_calculator as matcalc
import my_fits_package as myfits
import os

from deformable_mirror import DeformableMirror as DM

class SegmentedMirror(DM):
    """ Class defining the segmented deformable mirror """

    def __init__(self, mirror_geometry):
        
        # Define geometric quantities
        self.geom = mirror_geometry
        
        # Define mask and shape
        self.mask = self.geom.global_mask
        
        # Initialize segmented array
        self._define_segment_array()
        self._initialize_actuator_coordinates()
        
        
    def apply_masked_flat(self, mask, offset = None, slaving_mode:str = None):
        """
        Computes and applies the flat command
        for the current segment shape minus a given offset

        Parameters
        ----------
        mask : ndarray(bool)
            The subaperture mask over which to apply the flat
        
        offset : ndarray(float) [Npix], optional
            The shape offset from which to compute the flat.
            The default is no offset.
            
        Returns
        -------
        flat_rms : float
            The standard deviation of the obtained (masked) shape.

        """
        delta_shape = -self.surface
        
        if offset is not None:
            delta_shape += offset
            
        new_mask = np.logical_or(self.mask, mask)
        flat_ids = np.zeros(np.size(self.mask), dtype=int)
        
        pix_ids = self.valid_ids.flatten()
        flat_ids[pix_ids] = np.arange(np.sum(1-self.mask))
        masked_ids = flat_ids[~new_mask.flatten()]
        masked_shape = np.zeros_like(delta_shape)
        masked_shape[masked_ids] = delta_shape[masked_ids]
        
        for k,segment in enumerate(self.segment):
            segment_ids = self.valid_ids[k]
            segment_mask = segment.mask
            flat_mask = np.ones(np.size(segment_mask))
            flat_mask[~segment_mask.flatten()] = mask.flatten()[segment_ids]
            hex_mask = np.reshape(flat_mask, np.shape(segment_mask))
            local_mask = np.logical_or(segment.mask, hex_mask)
            
            if np.sum(1-local_mask) == len(segment_ids):
                segment.apply_flat()
                
            elif np.any(1-local_mask):
                cmd = segment.masked_flat_cmd(hex_mask, slaving=slaving_mode)
                segment.set_position(cmd)
        
        res_shape = self.surface[masked_ids]
        flat_rms = np.std(res_shape)
        
        if offset is not None:
            flat_rms = np.std(res_shape - offset)
        
        return flat_rms
    
    
    def segment_scramble(self, mode_amp = 10e-6, reset_shape:bool = False):
        """
        Applies a random shape to all segments using
        a random linear combination of Zernike modes,
        scaled by the inverse of the Noll number

        Parameters
        ----------
        mode_amp : float, optional
            Amplitude of the segment scramble. The default is 10e-6.
            
        reset_shape : bool, optional
            Sets the dsm shape to the scrambel and resets the actuator position
            The default is False.

        Returns
        -------
        None.

        """
        file_name = os.path.join(self.geom.savepath, 'initial_segment_scramble.fits')
        
        try:
            flat_img = myfits.read_fits(file_name)
        except FileNotFoundError:
            Nsegments = self.geom.n_hex
            ZMat = self.ZM
            
            # Retrieve number of modes from the interaction matrix
            n_modes = np.shape(ZMat)[-1]
            
            # Generate random mode coefficients
            mode_vec = np.random.randn(Nsegments*n_modes)
            
            # Probability inversely proportional to spatial frequency
            m = int(np.ceil((np.sqrt(8*n_modes)-1.)/2.))
            freq_vec = np.repeat(np.arange(m)+1,np.arange(m)+1)
            prob_vec = 1./freq_vec[:n_modes]
            prob_vec_rep = np.tile(prob_vec,Nsegments)
            
            # Modulate on the probability
            mode_vec = mode_vec * prob_vec_rep
            
            # Amplitude
            mode_vec *= mode_amp
            
            # Matrix product
            flat_img = matcalc.matmul(ZMat,mode_vec)
            myfits.write_to_fits(flat_img, file_name)
        
        if reset_shape:
            self.surface += flat_img - self.surface
            self.act_pos -= self.act_pos
        else:
            self.acquire_map(flat_img, plt_title = 'Segment scramble')
        
        
    def compute_global_zern_matrix(self, n_modes):
        """ Computes or reads the globl Zernike modes interaction matrix """
    
        file_path = os.path.join(self.geom.savepath, str(n_modes) + 'modes_global_zernike_mat.fits')
        
        try:
            self.glob_ZM = myfits.read_fits(file_path)
            return
        except FileNotFoundError:
            mask = self.mask.copy()
            scrambled_ZM = matcalc.compute_zernike_matrix(mask, n_modes)
            ids = np.zeros(np.size(mask),dtype=int)
            ids[~mask.flatten()] = np.arange(np.sum(1-mask))
            scrambled_ids = ids[self.valid_ids.flatten()]
            self.glob_ZM = scrambled_ZM[scrambled_ids,:]
            myfits.write_to_fits(self.glob_ZM, file_path)
            
        
    def compute_local_zern_matrix(self, n_modes):
        """ Computes or reads the local Zernike modes interaction matrix
        distributing the result to all segments and assemling the global
        sparse interaction matrix """
        
        file_path = os.path.join(self.geom.savepath, str(n_modes) + 'modes_local_zernike_mat.fits')
        
        try:
            ZM = myfits.read_fits(file_path)
        except FileNotFoundError:
            loc_ZM = matcalc.compute_zernike_matrix(self.geom.local_mask, n_modes)
            ZM = np.tile(loc_ZM,(self.geom.n_hex,1,1))
            myfits.write_to_fits(ZM, file_path)
        
        self.ZM = ZM
        for k,segment in enumerate(self.segment):
            segment.ZM = self.ZM[k]
            
        
    def initialize_IFF_and_R_matrices(self, simulate:bool = True):
        """ Initializes the local IFF and R matrices (same for all segments)
        distributing the result to all segments and assemling the global
        sparse IFF and R matrices """
        
        self.IFF_path = os.path.join(self.geom.savepath, 'global_influence_functions_matrix.fits')
        self.R_path = os.path.join(self.geom.savepath, 'global_reconstructor_matrix.fits')
        
        # Read/compute IFF and R
        try:
            self.IFF = myfits.read_fits(self.IFF_path)
            self.R = myfits.read_fits(self.R_path)
            
        except FileNotFoundError:
            IFF_cube = self._compute_local_IFF_cube(self.geom.local_act_coords)
            
            loc_IFF = matcalc.cube2mat(IFF_cube)
            self.IFF = np.tile(loc_IFF,(self.geom.n_hex,1,1))
            myfits.write_to_fits(self.IFF, self.IFF_path)
            
            loc_R = matcalc.compute_reconstructor(loc_IFF)
            self.R = np.tile(loc_R,(self.geom.n_hex,1,1))
            myfits.write_to_fits(self.R, self.R_path)
    
        # Distribute to all segments
        for k,segment in enumerate(self.segment):
            segment.IFF = self.IFF[k]
            segment.R = self.R[k]
            
            
    def _compute_local_IFF_cube(self, ref_act_coords, segment_id = None, simulate:bool = True):
        """ Reads or computes the IFF image cube for the actuator
        coordinates of the segment with given segment_id"""
        
        if segment_id is None:
            file_path = os.path.join(self.geom.savepath, 'Reference_IFF_image_cube.fits')
            try:
                IFF_cube = myfits.read_fits(file_path, is_ma = True)
                return IFF_cube
            except FileNotFoundError:
                pass   
        
        if simulate:
            IFF_cube = matcalc.simulate_influence_functions(ref_act_coords, self.geom.local_mask, self.geom.pix_scale)
        else:
            IFF_cube = matcalc.calculate_influence_functions(ref_act_coords, self.geom.local_mask, self.geom.mech_par)
       
        if segment_id is not None:
            for k,idx in enumerate(segment_id):
                file_path = os.path.join(self.geom.savepath, 'segment' + str(idx) + '_IFF_image_cube.fits')
                myfits.write_to_fits(IFF_cube, file_path)
            
        return IFF_cube
            
            
    def update_act_coords(self, segment_ids, new_act_coords, do_save:bool = True):
        """ Updates the actuator coordinates of the segment with given
        segment_id, updating the IFF and R matrices accordingly """
        
        n_acts = int(len(self.act_coords)/self.geom.n_hex)
        n_new = len(new_act_coords) 
        
        if n_new > n_acts:
            raise NotImplementedError('Current number of actuators is ' + str(n_acts) + 
                                      ' an increase to ' + str(n_new) + ' is not yet implemented')
        elif n_new < n_acts:
            nan_padding = np.empty([n_acts - n_new,2])
            nan_padding.fill(np.nan)
            new_act_coords = np.vstack([new_act_coords, nan_padding])
            
        # Compute IFF and R matrices
        IFF_cube = self._compute_local_IFF_cube(new_act_coords, segment_ids, simulate=True)
        loc_IFF = matcalc.cube2mat(IFF_cube)
        loc_R = matcalc.compute_reconstructor(loc_IFF)
            
        for k,segm_id in enumerate(segment_ids):
            act_ids = np.arange(segm_id*n_acts,(segm_id+1)*n_acts)
            seg = self.segment[segm_id]
            
            # Update actuator coordinates
            old_act_coords = seg.act_coords# - self.segment[segment_id].center
            seg.act_coords += new_act_coords - old_act_coords
            self.act_coords[act_ids] = new_act_coords + seg.center
            
            # Update IFF and R matrices
            # using += in order to update global at the same time
            old_IFF = seg.IFF
            seg.IFF += loc_IFF - old_IFF 
            old_R = seg.R
            seg.R += loc_R - old_R
            seg.n_acts += n_new - seg.n_acts
            
            # Reset shape and position
            seg.surface *= 0
            seg.act_pos *= 0
        
        # Save to .fits
        if do_save:
            myfits.write_to_fits(self.IFF, self.IFF_path)
            myfits.write_to_fits(self.R, self.R_path)
            myfits.write_to_fits(self.coords_path, self.act_coords) #np.save(self.coords_path, self.act_coords) 
            
            
    def _define_segment_array(self):
        """ Builds an array of Segment class objects,
        containing their center coordinates and the
        actuator positions """
        
        loc_mask = self.geom.local_mask
        
        local_ids = np.arange(np.size(loc_mask))
        local_valid_ids = local_ids[~loc_mask.flatten()]
        
        self.valid_ids = self.geom.valid_ids
        
        # Initialize segment array
        self.segment = []
        
        for k,coords in enumerate(self.geom.hex_centers):
            self.segment.append(Segment(coords, self.geom.local_mask, local_valid_ids))
            
            
    def _initialize_actuator_coordinates(self):
        """ Initializes the local actuator coordinates 
        for all segments in the mirror """
        
        self.geom.local_act_coords = self.geom.initialize_segment_act_coords()
        n_acts = len(self.geom.local_act_coords)
        n_hex = self.geom.n_hex
        n_pix = np.sum(1-self.geom.local_mask)
        
        self.coords_path = os.path.join(self.geom.savepath, 'global_actuator_coordinates.fits')
        try:
            self.act_coords = myfits.read_fits(self.coords_path) #np.load(self.coords_path + '.npy') #
        except FileNotFoundError:
            self.act_coords = np.tile(self.geom.local_act_coords,(n_hex,1))
            self.act_coords += np.tile(self.geom.hex_centers,n_acts).reshape([n_hex*n_acts,2])
        
        self.surface = np.zeros(np.sum(1-self.mask))
        self.act_pos = np.zeros(len(self.act_coords))
        self.n_acts = (np.ones(n_hex)*n_acts).astype(int)
        
        for k, segment in enumerate(self.segment):
            segment.act_coords = self.act_coords[n_acts*k:n_acts*(k+1),:] #- segment.center
            segment.act_pix_coords = matcalc.get_pixel_coords(segment.mask, self.geom.pix_scale, segment.act_coords - segment.center)
            segment.act_pos = self.act_pos[n_acts*k:n_acts*(k+1)]
            segment.surface = self.surface[n_pix*k:n_pix*(k+1)]
            segment.n_acts = (self.n_acts[k]).astype(int)
            
        # Save cooeds to fits
        myfits.write_to_fits(self.act_coords, self.coords_path) #np.save(self.coords_path, self.act_coords) # 

        
        