import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

from matrix_calculator import matmul

class DeformableMirror():
    """ 
    This class defines the methods for a generic deformable mirror.
    It implements the following functions:
        
        - acquire_map(): plot the current shape or a shape in input
        
        - get_position(): read and plot the actuator position at the actuator coordinates

        - set_position(): apply a zonal or modal command to the DM
        
        - compute_flat(): computes the flat command of the current shape with an (optional) offset
        
        - apply_flat(): applies the flat command of the current shape with an (optional) offset
    """
    
    def __init__(self):
        pass


    def acquire_map(self, surf2plot = None, plt_title:str = '', plt_mask = None):
        """
        Plots surf2plot or (default) the segment's
        current shape on the DM mask

        Parameters
        ----------
        surf2plot : ndarray(float) [Npix], optional
            The shape to plot. Defaults to the DM current shape.
            
        plt_title : str, optional
            The plot title. The default is no title.
            
        plt_mask : ndarray(bool), optional
            The mask to use on the plot. Default is None.

        """
        
        if surf2plot is None:
            surf2plot = self.surface
        
        pix_ids = self.valid_ids.flatten()
            
        image = np.zeros(np.size(self.mask))
        image[pix_ids] = surf2plot
        image = np.reshape(image, np.shape(self.mask))
        
        if plt_mask is None:
            plt_mask = self.mask
        else:
            plt_mask = np.logical_or(self.mask, plt_mask)
        
        image = np.ma.masked_array(image, plt_mask)
        
        plt.figure(figsize=(10,10))  
        plt.imshow(image, origin = 'lower', cmap = 'hot')
        
        img_rms = np.std(image.data[~image.mask])
        
        if plt_title is None:
            plt_title = f"RMS: {img_rms:.2e}"
            
        plt.axis('off')
        plt.title(plt_title)
        
        return img_rms
    
        

    def get_position(self):
        """
        Reads and plots the current actuators' 
        position at the actuators' coordinates 

        Returns
        -------
        pos : ndarray(float) [Nacts,]
            Array of current actuator positions on the segment.

        """
        
        pos = self.act_pos.copy()
        x,y = self.act_coords[0], self.act_coords[1] 
        
        act_pix_size = 3
        if np.sum(self.n_acts) < 100:
            act_pix_size = 12
        
        plt.figure(figsize = (10,10))
        plt.scatter(x,y, c=pos, s=act_pix_size**2, cmap='hot')
        plt.axis('equal')
        plt.colorbar()
        
        return pos
    
    
    def set_position(self, cmd_amps, absolute:bool = False, modal:bool = False):
        """
        Computes and applies a zonal/modal amplitude
        command to the segment

        Parameters
        ----------
        cmd_amps : ndarray(float) [Nacts]
            The array of zonal (modal) amplitudes.
            
        absolute : bool, optional
            True if the command is absolute and not
            relative to the current actuators' position.
            The default is False.
            
        modal : bool, optional
            True for modal amplitudes. The default is False.

        """
        
        if modal: # convert modal to zonal
            
            # length check
            mode_amps = cmd_amps.copy()
            n_modes = np.shape(self.ZM)[1]
            if len(mode_amps) < n_modes:
                mode_amps = np.zeros(n_modes)
                mode_amps[:len(cmd_amps)] = cmd_amps
        
            shape = matmul(self.ZM, mode_amps)
            cmd_amps = matmul(self.R, shape)
        
        # Position (zonal) command
        if absolute:
            cmd_amps -= self.act_pos

        # Update positions and shape
        self.act_pos += cmd_amps
        self.surface += matmul(self.IFF, cmd_amps)

    
    def compute_PSF(self, pupil_mask = None, surface = None, oversampling: int = 4):
        """ Computes the mirror PSF """

        if pupil_mask is None:
            pupil_mask = self.mask.copy()

        mask = np.logical_or(pupil_mask, self.mask)

        if surface is None:
            surface = self.surface

        pix_ids = self.valid_ids.flatten()
            
        image = np.zeros(np.size(mask))
        image[pix_ids] = surface
        phase = np.reshape(image, np.shape(mask))

        nx,ny = np.shape(mask)
        npix = np.max((nx,ny))
        x_pad = (npix*oversampling-nx)//2
        y_pad = (npix*oversampling-ny)//2

        pad_phase = np.pad(phase, ((x_pad,x_pad),(y_pad,y_pad)))
        pad_mask = np.pad(mask, ((x_pad,x_pad),(y_pad,y_pad)), 'constant', constant_values=1)
        pad_img = np.ma.masked_array(np.exp(1j*pad_phase), pad_mask)

        psf = np.fft.fftshift(np.fft.fft2(pad_img))
        # psf *= 1/np.sum(psf)

        return psf 
    
    
    # def compute_PSD(self, surface = None, oversampling:int = 4, pix_scale:float=1):
    #     """ Computes the mirror PSD """

    #     from arte.utils.radial_profile import computeRadialProfile

    #     psf = self.compute_PSF(surface, oversampling)
    #     power = np.abs(psf**2)

    #     nx,ny = np.shape(self.mask)[0]*oversampling, np.shape(self.mask)[1]*oversampling
    #     psd, pix_distance = computeRadialProfile(power, centerInPxX=nx//2, centerInPxY=ny//2)
    #     rad_distance = pix_distance/pix_scale

    #     return psd, rad_distance
        
    
    def compute_flat(self, offset = None):
        """
        Computes the flat command for the current
        mirror shape minus a given offset

        Parameters
        ----------
        offset : ndarray(float) [Npix], optional
            The shape offset from which to compute the flat.
            The default is no offset.

        Returns
        -------
        act_cmd : ndarray(float) [Nacts]
            The vector of actuator commands.

        """
        
        delta_shape = -self.surface
        
        if offset is not None:
            delta_shape += offset
            
        act_cmd = matmul(self.R, delta_shape)
        
        self._flat_cmd = act_cmd
        
        return act_cmd
        
    
    
    def apply_flat(self, offset = None):
        """
        Applies the flat command minus a given offset

        Parameters
        ----------
        offset : ndarray(float) [Npix], optional
            The shape offset from which to compute the flat.
            The default is no offset.

        Returns
        -------
        flat_rms : float
            The standard deviation of the obtained shape.

        """
        
        if self._flat_cmd is None:
            self.compute_flat(offset)
            
        self.set_position(self._flat_cmd)
        res_shape = self.surface
            
        flat_rms = np.std(res_shape)
        
        if offset is not None:
            flat_rms = np.std(res_shape - offset)
        
        return flat_rms

