import numpy as np
from deformable_mirror import DeformableMirror as DM
from matrix_calculator import matmul

class Segment(DM):
    """ Class defining the single segment
    with center coordinates and actuator displacements """
    
    def __init__(self, center_coordinates, mask, valid_ids):
        
        self.center = center_coordinates
        self.mask = mask
        self.valid_ids = valid_ids
        self._flat_cmd = None
        
        
    def masked_flat_cmd(self, mask, offset = None, slaving:str = None):
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
            
        slaving : str, optional
            The way to treat slave actuators.
                'zero'    : set slave actuator command to zero 
                'interp'  : thin plate spline interpolation
                'exclude' : exclude slaves from the reconstructor computation
            
        Returns
        -------
        flat_rms : float
            The standard deviation of the obtained (masked) shape.

        """
        delta_surface = -self.surface
        
        if offset is not None:
            delta_surface += offset
            
        new_mask = np.logical_or(self.mask, mask)
        flat_ids = np.zeros(np.size(self.mask), dtype=int)
        
        pix_ids = self.valid_ids.flatten()
        flat_ids[pix_ids] = np.arange(np.sum(1-self.mask))
        valid_ids = flat_ids[~new_mask.flatten()]
        masked_shape = np.zeros_like(delta_surface)
        masked_shape = delta_surface[valid_ids]

        masked_IFF = self.IFF[valid_ids,:]
        masked_R = np.linalg.pinv(masked_IFF)
        act_cmd = matmul(masked_R, masked_shape)
        
        # Actuator slaving
        if slaving is not None:
            visible_acts = self._masked_act_ids(new_mask)
            
            if slaving == 'zero':
                pad_cmd = np.zeros_like(act_cmd)
                pad_cmd[visible_acts] = act_cmd[visible_acts]
                act_cmd = pad_cmd
                
            elif slaving == 'interp':
                from tps import ThinPlateSpline
                tps = ThinPlateSpline(alpha=0.0)
                tps.fit(self.act_coords[visible_acts], act_cmd[visible_acts])
                rescaled_cmd = tps.transform(self.act_coords)
                act_cmd = rescaled_cmd[:,0]
                
            elif slaving == 'exclude':
                masked_IFF = self.IFF[valid_ids,:]
                masked_IFF = masked_IFF[:,visible_acts]
                masked_R = np.linalg.pinv(masked_IFF)
                
                pad_cmd = np.zeros_like(act_cmd)
                pad_cmd[visible_acts] = matmul(masked_R, masked_shape)
                act_cmd = pad_cmd
                
            else:
                raise ValueError("Incorrect input slaving mode! Available modes are: 'interp', 'zero', 'exclude'")
        
        return act_cmd
    
    
        
    def _masked_act_ids(self, mask):
        """ Finds the ids of the actuators visible
        on the mask """
        
        flat_ids = np.zeros(np.size(mask),dtype=int)
        pix_ids = self.valid_ids.flatten()
        flat_ids[pix_ids] = np.arange(np.sum(1-self.mask))
        valid_ids = flat_ids[~mask.flatten()]
        
        X,Y = np.shape(mask)
        pix_coords = np.zeros([X*Y,2])
        pix_coords[:,0] = np.repeat(np.arange(X),Y)
        pix_coords[:,1] = np.tile(np.arange(Y),X)
        pix_coords = (pix_coords[pix_ids]).astype(int)
        
        visible_acts = []
        for k in range(self.n_acts):
            if np.min(np.sum(np.abs(pix_coords[valid_ids]-self.act_pix_coords[k]),axis=1)) == 0:
                visible_acts.append(k)
        visible_acts = np.array(visible_acts,dtype=int)
        
        return visible_acts
        
        
        
        
    
    
    
        