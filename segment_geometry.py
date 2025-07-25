import os
import numpy as np

from read_configuration import read_config
from rotate_coordinates import cw_rotate
import my_fits_package as myfits


# Some useful variables and functions
SIN60 = np.sin(np.pi/3.)
COS60 = np.cos(np.pi/3.)

def sum_n(n):
    return int(n*(n+1)/2)

def n_hexagons(n_rings):
    return int(1 + (6 + n_rings*6)*n_rings/2)


def circular_mask(radius: float, pix_scale, mask_shape, center=np.array([0.,0.])):
    """
    Creates a circular mask of given dimensions

    Parameters
    ----------
    radius : float
        The mask radius in meters.
        
    pix_scale : float
        The number of pixels per meter.
        
    mask_shape : ndarray(int)
        The x,y mask pixel dimensions.
        
    center : ndarray(float), optional
        The mask center coordinates in meters.
        The default is the origin.

    Returns
    -------
    circ_mask : ndarray(bool)
        The obtained circular mask.

    """
    X,Y = mask_shape
    r_pix = radius * pix_scale
    c_pix = center * pix_scale + np.array([Y/2,X/2])
    
    dist = lambda x,y: np.sqrt(x**2+y**2)
    
    circ_mask = np.fromfunction(lambda i,j: dist(i-c_pix[1],j-c_pix[0]) > r_pix, [X,Y])
    circ_mask = (circ_mask).astype(bool)
    
    return circ_mask
    

class HexagonGeometry():
    
    def __init__(self, TN):
        
        # Read configuration files
        dm_par, opt_par, mech_par, save_path = read_config(TN)
        
        self.gap = dm_par[0]
        self.hex_side_len = dm_par[1]
        self.n_rings = int(dm_par[2])
        self.act_pitch = dm_par[3]
        self.act_radius = dm_par[4]
        self.center_bool = bool(dm_par[5])

        self.pix_scale = opt_par[0]
        self.opt_r = opt_par[1]
        self.opt_x = opt_par[2]
        self.opt_y = opt_par[3]

        self.mech_par = mech_par
        
        self.savepath = save_path
        self.n_hex = n_hexagons(self.n_rings)
        
        # h = (self.gap + 2.*self.hex_side_len*SIN60)*(self.n_rings+1)/2. 
        # d = (self.gap + self.hex_side_len + self.hex_side_len*COS60)*self.n_rings - self.hex_side_len/2.
        # R = np.sqrt(h**2+d**2) # inscribed circle radius
        
        try: # Create new folder
            os.mkdir(save_path)
        except FileExistsError: # Folder already existing
            pass
        
        self._define_local_mask()
        self._define_segment_centers()
        self._assemble_global_mask()
        self._assemble_optical_mask()
        self._define_hex_outline() # plotting only
        
        
    def initialize_segment_act_coords(self):
        """
        Defines the local actuator coordinates 
        on the hexagonal the segment 

        Returns
        -------
        local_act_coords : ndarray [Nacts,2]
            The x,y cordinates of the actuators on the hexagonal segment.

        """
    
        file_path = os.path.join(self.savepath, 'local_act_coords.fits') #'.fits'
        try:
            local_act_coords = myfits.read_fits(file_path) # np.load(file_path + '.npy')
            return local_act_coords
        except FileNotFoundError:
            pass
        
        # Normalize quantities by hexagon side length (hex_side_len)
        L =self.hex_side_len
        rad = self.act_radius/L
        pitch = self.act_pitch/L
        
        acts_per_side = (1+pitch)/(2*rad + pitch) 
        dx = 2*rad+pitch
        
        acts_per_side = int(acts_per_side-1)
        n_acts_tri = sum_n(acts_per_side)
        
        act_coords = np.zeros([2,n_acts_tri*6+1])
        
        for k in range(acts_per_side):
            y = np.linspace(SIN60*k*dx,0.,k+1)
            x = (k+1)*dx - COS60/SIN60 * y
            n = k+1
            p = np.zeros([2,6*n])
            p[0,0:n] = x
            p[1,0:n] = y
            p[:,n:2*n] = cw_rotate(p[:,0:n],np.array([np.pi/3]))
            p[:,2*n:3*n] = cw_rotate(p[:,n:2*n],np.array([np.pi/3]))
            p[:,3*n:] = cw_rotate(p[:,0:3*n],np.array([np.pi]))
            act_coords[:,1+sum_n(k)*6:1+sum_n(k+1)*6] = p
            
        # Rescaling
        local_act_coords = act_coords*self.hex_side_len
            
        # Save result
        myfits.write_to_fits(local_act_coords, file_path) #np.save(file_path, local_act_coords) #
        
        return local_act_coords


    def _define_local_mask(self):
        """ Forms the hexagonal mask to be placed 
        at the given hexagonal coordinates """
    
        file_path = os.path.join(self.savepath, 'local_mask.fits')
        try:
            self.local_mask = myfits.read_fits(file_path, is_bool = True)
            return 
        except FileNotFoundError:
            pass
    
        L = self.hex_side_len*self.pix_scale
    
        # Consider points in the upper-left of the hexagon
        max_y = int(L*SIN60)
        max_x = int(L/2.+L*COS60)
    
        mask_ul = np.fromfunction(lambda i,j: j >= L/2. + i/SIN60*COS60, [max_y,max_x])
    
        mask = np.zeros([2*max_y,2*max_x], dtype=bool)
    
        mask[0:max_y,max_x:] = mask_ul # upper left
        mask[0:max_y,0:max_x] = np.flip(mask_ul,1) # upper right
        mask[max_y:,:] = np.flip(mask[0:max_y,:],0) # lower
    
        # Save as private variable and to .fits
        self.local_mask = mask
        myfits.write_to_fits((mask).astype(np.uint8), file_path)
        
          
    def _define_segment_centers(self):
        """ Defines and saves the coordinates of the 
        centers of all hexagonal segments """
        
        file_path = os.path.join(self.savepath, 'hex_centers_coords.fits')
        try:
            self.hex_centers = myfits.read_fits(file_path) # np.load(file_path + '.npy')
            if self.center_bool is False: 
                self.n_hex -= 1
            return
        except FileNotFoundError:
            pass
        
        # Number of hexes
        hex_centers = np.zeros([2,self.n_hex])
        
        # Height of hex + gap
        L = self.gap + 2.*self.hex_side_len*SIN60
        angles = np.pi/3. * (np.arange(5)+1)
        
        ring_vec = np.arange(self.n_rings)
        
        for ring_ctr in ring_vec:
            
            hex_ctr = n_hexagons(ring_ctr)
            ring_ctr += 1
            R = L*ring_ctr
            
            aux = np.array([R*SIN60,R*COS60])
            hex_centers[:,hex_ctr] = aux
            hex_centers[:,hex_ctr+ring_ctr:hex_ctr+6*ring_ctr:ring_ctr] = cw_rotate(aux, angles)
            
            if ring_ctr > 1:
                for j in range(ring_ctr-1):
                    shift = self.gap + 2.*self.hex_side_len*SIN60
                    aux[0] = hex_centers[0,hex_ctr] 
                    aux[1] = hex_centers[1,hex_ctr] - (j+1)*shift
                    hex_centers[:,hex_ctr+j+1] = aux
                    hex_centers[:,hex_ctr+j+1+ring_ctr:hex_ctr+j+1+6*ring_ctr:ring_ctr] = cw_rotate(aux, angles)
                    
        if self.center_bool is False: # remove center segment
            hex_centers = hex_centers[:,1:]
            self.n_hex -= 1
    
        # Save as private variable and to .fits
        self.hex_centers = hex_centers
        myfits.write_to_fits(hex_centers, file_path) #np.save(file_path, hex_centers) #
        
    
    def _assemble_global_mask(self):
        """ Assemble the global segmented mask """
        
        ids_file_path = os.path.join(self.savepath, 'valid_ids.fits') #os.path.join(self.savepath, 'valid_ids')
        file_path = os.path.join(self.savepath, 'global_mask.fits')
        try:
            self.global_mask = myfits.read_fits(file_path, is_bool=True)
            self.valid_ids = myfits.read_fits(ids_file_path) # np.load(ids_file_path + '.npy') 
            return
        except FileNotFoundError:
            pass
        
        # Height of hex + gap
        L = self.gap + 2.*self.hex_side_len*SIN60
        
        # Full mask dimensions
        Ny = np.ceil((L*self.n_rings +self.hex_side_len*SIN60)*2*self.pix_scale)
        Nx = np.ceil((L*self.n_rings*SIN60 +self.hex_side_len*(0.5+COS60))*2*self.pix_scale)
        
        # Hexagon centers pixel coordinates
        pix_coords = self.hex_centers*self.pix_scale + np.array([[Nx],[Ny]])/2.
        
        My,Mx = np.shape(self.local_mask)
        x = np.arange(Mx,dtype=int)
        y = np.arange(My,dtype=int)
        X,Y = np.meshgrid(x,y)
        local_X = X[~self.local_mask]
        local_Y = Y[~self.local_mask]
        local_row_idx = local_Y - int(My/2)
        local_col_idx = local_X - int(Mx/2)
    
        rep_local_row = np.tile(local_row_idx,self.n_hex)
        rep_local_col = np.tile(local_col_idx,self.n_hex)
        
        hex_data_len = np.sum(1-self.local_mask)
        rep_pix_coords = np.repeat(pix_coords, hex_data_len, axis = 1)
        
        global_row_idx = (rep_local_row + rep_pix_coords[1,:]).astype(int)
        global_col_idx = (rep_local_col + rep_pix_coords[0,:]).astype(int)
        
        row_ids = np.reshape(global_row_idx,[self.n_hex,hex_data_len])
        col_ids = np.reshape(global_col_idx,[self.n_hex,hex_data_len])
        
        # Data
        data = np.ones([self.n_hex,hex_data_len], dtype=bool)
        
        # Mask definition
        global_mask = np.ones([np.max(row_ids)+1,np.max(col_ids)+1])
        global_mask[row_ids,col_ids] -= data
        
        # Save as private variable and to .fits
        self.global_mask = (global_mask).astype(bool)
        myfits.write_to_fits((global_mask).astype(np.uint8), file_path)
        
        # Save valid hexagon indices
        valid_ids = (row_ids*np.shape(global_mask)[1] + col_ids).astype(int)
        
        # Save as private variable and to .fits
        self.valid_ids = valid_ids
        myfits.write_to_fits(valid_ids, ids_file_path) # np.save(ids_file_path, valid_ids) #
        
        
    def _assemble_optical_mask(self):
        """ Assemble the optical mask """
        
        file_path = os.path.join(self.savepath, 'optical_mask.fits')
        try:
            self.optical_mask = myfits.read_fits(file_path, is_bool=True)
            return
        except FileNotFoundError:
            pass
        
        # mask_x, mask_y = np.shape(self.global_mask)
        
        # R_pix = np.ceil(self.opt_r*self.pix_scale).astype(int)
        # X_pix = np.ceil(self.opt_x*self.pix_scale).astype(int) + mask_y/2
        # Y_pix = np.ceil(self.opt_y*self.pix_scale).astype(int) + mask_x/2
    
        # circ_mask = np.fromfunction(lambda i,j: np.sqrt((j - X_pix)**2+(i - Y_pix)**2) >= R_pix, np.shape(self.global_mask))
    
        # self.optical_mask = (circ_mask).astype(bool)
        
        self.optical_mask = circular_mask(self.opt_r, self.pix_scale,
                                          np.shape(self.global_mask),
                                          np.array([self.opt_x,self.opt_y]))
        
        myfits.write_to_fits((self.optical_mask).astype(np.uint8), file_path)
        
        
    def _define_hex_outline(self):
        """ Defines the point coordinates of the hex vertices"""
        
        x_hex = np.array([-0.5-COS60,-0.5,0.5,0.5+COS60,np.nan])
        y_hex_p = np.array([0.,SIN60,SIN60,0.,np.nan])
        y_hex_m = np.array([0.,-SIN60,-SIN60,0.,np.nan])
        c_hex = np.vstack( (np.tile(x_hex,(1,2)),np.hstack((y_hex_m,y_hex_p)) ) )
        c_hex  *= self.hex_side_len
        
        self.hex_outline = c_hex