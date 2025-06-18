import numpy as np
from math import factorial as fac
# import matplotlib.pyplot as plt

def fit_zernike(noll_ids, img:np.ma.masked_array, scale_length:float = None):
    """
    Fits the given image with Zernike polynomials of the given indices

    Parameters
    ----------
    noll_ids : ndarray(int) [Nzern,]
        Array of Noll indices to fit.
        
    img : np.ma.masked_array
        The image to fit.
        
    scale_length : float, optional
        The scale length to use for the Zernike fit.
        The default is the maximum of the image mask shape.

    Returns
    -------
    zern_coeffs : ndarray(float) [Nzern,]
        The array of fitted zernike coefficients.
        
    zern_img : ndarray(float) [Npix,]
        The flattened imaged obtained from the fit.

    """
    
    # Image mask
    img_mask = img.mask
    img_data = img.data[~img_mask]
    
    ZernMat = generate_zernike_matrix(noll_ids, img_mask, scale_length)
        
    zern_coeffs = np.linalg.pinv(ZernMat) @ img_data
    zern_img = ZernMat @ zern_coeffs

    return zern_coeffs, zern_img


def generate_zernike_matrix(noll_ids, img_mask, scale_length:float = None):
    """
    Generates the interaction matrix of the Zernike modes with Noll index
    in noll_ids on the mask in input

    Parameters
    ----------
    noll_ids : ndarray(int) [Nzern,]
        Array of Noll indices to fit.
        
    img_mask : matrix bool
        Mask of the desired image.
        
    scale_length : float, optional
        The scale length to use for the Zernike fit.
        The default is the maximum of the image mask shape.

    Returns
    -------
    ZernMat : ndarray(float) [Npix,Nzern]
        The Zernike interaction matrix of the given indices on the given mask.

    """
    
    n_pix = int(np.sum(1-img_mask))
    n_zern = len(noll_ids)
    ZernMat = np.zeros([n_pix,n_zern])
    
    for i in range(n_zern):
        ZernMat[:,i] = _project_zernike_on_mask(noll_ids[i], img_mask, scale_length)
        
    return ZernMat


def _project_zernike_on_mask(noll_number:int, mask, scale_length:float = None, mask_center = None):
    """
    Project the Zernike polynomials identified by the Noll number in input
    on a given mask.
    The polynomials are computed on the circle inscribed in the mask by default,
    or on a circle of radius scale_length if the corresponding input is given
    Masked data is then normalized as follows:
    data = ma.data[~ma.mask], data = (data - mean(data))/std(data)

    Parameters
    ----------
    noll_number : int
        Noll index of the desired Zernike polynomial.
    mask : matrix bool
        Mask of the desired image.

    Returns
    -------
    masked_data : ndarray
        Flattenned array of the masked values of the Zernike 
        shape projected on the mask.

    """
    if noll_number < 1:
        raise ValueError("Noll index must be equal to or greater than 1")

    # Image dimensions
    X,Y = np.shape(mask)

    # Determine circle radius on to which define the Zernike
    if scale_length is not None:
        r = scale_length
    else:
        r = np.max([X,Y])/2

    if mask_center is None:
        yc = Y/2.
        xc = X/2.
    else:
        xc = mask_center[0]
        yc = mask_center[1]

    # Conversion to polar coordinates on circle of radius r 
    phi = lambda i,j: np.arctan2((j-yc)/r,(i-xc)/r)
    rho = lambda i,j: np.sqrt(((j-yc)/r)**2+((i-xc)/r)**2)
            
    mode = np.fromfunction(lambda i,j: _zernikel(noll_number, rho(i,j), phi(i,j)), [X,Y])

    # masked_mode = np.ma.masked_array(mode,mask)
    # plt.figure();plt.imshow(masked_mode,origin='lower');plt.title('Zernike ' + str(noll_number))
    # masked_data = masked_mode.data[~masked_mode.mask]
    
    #masked_data = mode[~mask]
    masked_data = mode.flatten()[mask.flatten()<1]

    # Normalization of the masked data: null mean and unit STD
    if noll_number > 1:
        masked_data = (masked_data - np.mean(masked_data))/np.std(masked_data)

    return masked_data
    



########### from M4SW ############

def _zernikel(j, rho, phi):
    """
    Calculate Zernike polynomial with Noll coordinate j given a grid of radial
    coordinates rho and azimuthal coordinates phi.

    >>> zernikel(0, 0.12345, 0.231)
    1.0
    >>> zernikel(1, 0.12345, 0.231)
    0.028264010304937772
    >>> zernikel(6, 0.12345, 0.231)
    0.0012019069816780774
    """
    m, n = _j2mn_noll(j)
    
    return _zernike(m, n, rho, phi)


def _zernike(m, n, rho, phi):
    """
    Calculate Zernike polynomial (m, n) given a grid of radial
    coordinates rho and azimuthal coordinates phi.

    >>> zernike(3,5, 0.12345, 1.0)
    0.0073082282475042991
    >>> zernike(1, 3, 0.333, 5.0)
    -0.15749545445076085
    """
    if (m > 0): return _zernike_rad(m, n, rho) * np.cos(m * phi)
    if (m < 0): return _zernike_rad(-m, n, rho) * np.sin(-m * phi)
    return _zernike_rad(0, n, rho)


def _zernike_rad(m, n, rho):
    """
    Calculate the radial component of Zernike polynomial (m, n)
    given a grid of radial coordinates rho.

    >>> zernike_rad(3, 3, 0.333)
    0.036926037000000009
    >>> zernike_rad(1, 3, 0.333)
    -0.55522188900000002
    >>> zernike_rad(3, 5, 0.12345)
    -0.007382104685237683
    """

    if (n < 0 or m < 0 or abs(m) > n):
        raise ValueError

    if ((n-m) % 2):
        return rho*0.0

    pre_fac = lambda k: (-1.0)**k * fac(n-k) / ( fac(k) * fac( int((n+m)/2.0 - k) ) * fac( int((n-m)/2.0 - k) ) )

    return sum(pre_fac(k) * rho**(n-2.0*k) for k in range((n-m)//2+1))


def _j2mn_noll(j):
    """
    Find the [n,m] list giving the radial order n and azimuthal order
    of the Zernike polynomial of Noll index j.

    Parameters:
        j (int): The Noll index for Zernike polynomials

    Returns:
        list: n, m values
    """
    n = int((-1.+np.sqrt(8*(j-1)+1))/2.)
    p = (j-(n*(n+1))/2.)
    k = n%2
    m = int((p+k)/2.)*2 - k

    if m!=0:
        if j%2==0:
            s=1
        else:
            s=-1
        m *= s

    return [m, n]

# def _j2mn_ansi(j):
#     """
#     Find the [n,m] list giving the radial order n and azimuthal order
#     of the Zernike polynomial of ANSI index j.

#     Parameters:
#         j (int): The ANSI index for Zernike polynomials

#     Returns:
#         list: n, m values
#     """
#     n = 0
#     while (j > n):
#         n += 1
#         j -= n

#     m = -n+2*j

#     return m, n
