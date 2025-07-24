from astropy.io import fits
import numpy as np

def write_to_fits(data_vec, file_path:str):
    """
    Simple function to save files in .fits format.
    If a list of masked arrays is passed in input,
    only the first mask is saved (see read_fits)

    Parameters
    ----------
    data_vec : ndarray
        Array of data to save. Can be a masked array.
    file_path : str
        The complete path where the file is saved.

    Returns
    -------
    None.

    """

    if isinstance(data_vec,list): # list

        if hasattr(data_vec[0],'mask'): # masked array
            img_mask = data_vec[0].mask
            fits.writeto(file_path, (img_mask).astype(np.uint8), overwrite=True)
            fits.append(file_path, data_vec[0].data)
        else:
            fits.writeto(file_path, data_vec[0], overwrite=True)

        for vec in data_vec[1:]:

            if hasattr(vec,'mask'): # masked array
                fits.append(file_path, vec.data)

            else:
                fits.append(file_path, vec)

    else:
        # Print to a .fits file
        if hasattr(data_vec,'mask'): # masked array
            fits.writeto(file_path, (data_vec.mask).astype(np.uint8), overwrite=True)
            fits.append(file_path, data_vec.data)
        else:
            fits.writeto(file_path, data_vec, overwrite=True)
            



# Reading files
def _read_fits_file(file_path:str, idx:int = 0, is_bool:bool = False, img_mask = None):
    """
    Reads a single fits file.

    Parameters
    ----------
    file_path : str
        The complete path where the file was saved.
    idx : int
        The list index of the hdu to be read.
    is_bool : bool
        Wheter the output is boolean (i.e. file is a mask) or not.
    img_mask : ndarray (bool), optional
        If not None, a mask is appended to the image . The default is None

    Returns
    -------
    data_out : ndarray
        The array of data read. If an img_mask is given as input
        the output is a masked array

    """

    hdu = fits.open(file_path)

    if is_bool: # Boolean array
        data_out = np.array(hdu[idx].data).astype(bool)

    elif img_mask is not None: # masked array
        img = np.array(hdu[idx].data)
        data_out = np.ma.masked_array(img, mask=img_mask)

    else: # default
        data_out = np.array(hdu[idx].data)

    return data_out



def read_fits(file_path:str, list_len:int = 1, is_bool:bool = False, is_ma:bool = False):
    """
    Reads a list of files of length list_len saved in file_path, 
    outputs the data read (in boolean if is_bool, as masked arrays if is_ma)

    Parameters
    ----------
    file_path : str
        The complete path where the data is saved.
    list_len : int, optional
        If the input is a list of arrays, this is the length of the list. The default is 1.
    is_bool : bool, optional
        Wheter the output is boolean (i.e. file is a mask) or not. The default is False.
    is_ma : bool, optional
        Wheter the output is a masked array or not. The default is False.

    Returns
    -------
    data_out : ndarray
        The array of data read.

    """

    data_out = []

    if is_ma:
        img_mask = _read_fits_file(file_path, idx = 0, is_bool = True)
    else:
        img_mask = None
        
    for k in range(list_len):
        out = _read_fits_file(file_path, k + is_ma, is_bool, img_mask)
        data_out.append(out)

    if list_len == 1:
        data_out = data_out[0]

    return data_out
