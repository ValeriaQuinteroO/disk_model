import numpy as np
import pdb
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import scipy.special as special
import scipy.fft as fft
import oitools

def mas2rad(x):
    import numpy as np
    y=x/3600.0/1000.0*np.pi/180.0
    y=np.array(y)
    return y

def Gaus(a, q):
    return np.exp(-(np.pi*a*q)**2/np.log(2))

def Lor(a, q):
    return np.exp(-2*np.pi*a*q/np.sqrt(3))

to_rd = lambda m, d: m * np.exp(1j * np.deg2rad(d))
to_pd = lambda x: (abs(x), np.rad2deg(np.angle(x)))

def compute_vis_matrix(im, u, v, scale):
    import numpy as np
    import math
    import pdb
    sz = np.shape(im)
    if sz[0] % 2 == 0:
        x, y = np.mgrid[-np.floor(sz[1] / 2 - 1):np.floor(sz[1] / 2):sz[1] * 1j,
               np.floor(sz[0] / 2 - 1):-np.floor(sz[0] / 2):sz[0] * 1j]
    else:
        x, y = np.mgrid[-np.floor(sz[1] / 2):np.floor(sz[1] / 2):sz[1] * 1j,
               np.floor(sz[0] / 2):-np.floor(sz[0] / 2):sz[0] * 1j]
    x = x * scale
    y = y * scale
    xx = x.reshape(-1)
    yy = y.reshape(-1)
    im = im.reshape(-1)
    arg = -2.0 * np.pi * (u * yy + v * xx)
    reales = im.dot(np.cos(arg))
    imaginarios = im.dot(np.sin(arg))

    return reales, imaginarios


def model_fourier3(ucoord, vcoord, theta1, inc1, c, s, la, lkr, flor, fs, fd, fo):
    #pdb.set_trace()
    Vis = np.zeros([ucoord.shape[0]], dtype='complex')

    ar = np.sqrt(10 ** (2 * la) / (1 + 10 ** (2 * lkr)))
    ak = ar * 10 ** (lkr)
    
    ar = mas2rad(ar)
    ak = mas2rad(ak)
    a = np.sqrt(ar ** 2 + ak ** 2)
    
    theta1 = 90 - theta1
    for i in range(ucoord.shape[0]):
        #### Disk Skeleton
        u_theta1 = -1.0*ucoord[i] * np.sin(np.deg2rad(theta1)) + vcoord[i] * np.cos(np.deg2rad(theta1))
        v_theta1 = ( ucoord[i] * np.cos(np.deg2rad(theta1)) + vcoord[i] * np.sin(np.deg2rad(theta1))) / np.cos(
            np.deg2rad(inc1))
       
        r_uvti1 = np.sqrt(u_theta1 ** 2 + v_theta1 ** 2)
        mod = c + 1j * s
        phasor = to_pd(mod)

        #mod2 = c2 + 1j * s2
        #phasor2 = to_pd(mod2)
        
        mod_uv = u_theta1 + 1j * v_theta1
        phasor_uv = to_pd(mod_uv)


        
        skeleton = special.j0(2 * np.pi * r_uvti1 * ar) + (-1j) ** (1) * phasor[0] * np.cos(
            1 * np.deg2rad((phasor_uv[1] - phasor[1]))) * \
                   special.j1(2 * np.pi * r_uvti1 * ar) #+ (-1j)**2 * phasor2[0] * np.cos(2*np.deg2rad((phasor_uv[1] - phasor2[1])))*special.jn(2,2*np.pi*r_uvti1*ar)

        #### Convolution kernel
        rr = np.sqrt(ucoord[i] ** 2 + vcoord[i] ** 2)
        Vc = (1 - flor) * Gaus(ak, r_uvti1) + flor * Lor(ak, r_uvti1)

        #### Total Visibility
        Vis[i] = (fs + fd * skeleton * Vc) / (fd + fs + fo)

    return Vis


def model_im3(im_size, scale, theta1, inc1, c, s, la, lkr, flor, fs, fd, fo, wave, year):
    sz = 1.0 * im_size
    if sz % 2 == 0:
        x, y = np.mgrid[-np.floor(sz / 2 - 1):np.floor(sz / 2):sz * 1j,
               np.floor(sz / 2 - 1):-np.floor(sz / 2):sz * 1j]
    else:
        x, y = np.mgrid[-np.floor(sz / 2):np.floor(sz / 2):sz * 1j,
               np.floor(sz / 2):-np.floor(sz / 2):sz * 1j]
    v = x * scale
    u = y * scale

    pyfits.writeto('ucoord.fits', u, overwrite=True)
    pyfits.writeto('vcoord.fits', v, overwrite=True)

    Vis = np.zeros([x.shape[0], x.shape[1]], dtype='complex')
    Vis2 = np.zeros([x.shape[0], x.shape[1]], dtype='complex')

    ar = np.sqrt(10 ** (2 * la) / (1 + 10 ** (2 * lkr)))
    ak = ar * 10 ** (lkr)
    ar = mas2rad(ar)
    ak = mas2rad(ak)
    a = np.sqrt(ar ** 2 + ak ** 2)
    
    theta1 = 90 - theta1
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            #### Disk Skeleton
            u_theta1 = -1.0 * u[i, j] * np.sin(np.deg2rad(theta1)) + v[i, j] * np.cos(np.deg2rad(theta1))
            v_theta1 = (u[i, j] * np.cos(np.deg2rad(theta1)) + v[i, j] * np.sin(np.deg2rad(theta1))) / np.cos(
                np.deg2rad(inc1))
            r_uvti1 = np.sqrt(u_theta1 ** 2 + v_theta1 ** 2)
            mod = c + 1j * s
            phasor = to_pd(mod)
            #mod2 = c2 + 1j * s2
            #phasor2 = to_pd(mod2)
            mod_uv = u_theta1 + 1j * v_theta1
        
            phasor_uv = to_pd(mod_uv)
            skeleton = special.j0(2 * np.pi * r_uvti1 * ar) + (-1j) ** (1) * phasor[0] * np.cos(
                1 * np.deg2rad((phasor_uv[1] - phasor[1]))) * \
                       special.j1(2 * np.pi * r_uvti1 * ar) #+ (-1j)**2 *phasor2[0] * np.cos(2*np.deg2rad((phasor_uv[1] - phasor2[1])))*special.jn(2,2*np.pi*r_uvti1*ar)

            #### Convolution kernel
            rr = np.sqrt(u[i, j] ** 2 + v[i, j] ** 2)
            Vc = (1 - flor) * Gaus(ak, r_uvti1) + flor * Lor(ak, r_uvti1)

            #### Total Visibility
            Vis[i, j] = (fs + fd * skeleton * Vc ) / (fd + fs + fo)
            Vis2[i, j] = (fs + fd * skeleton ) / (fs + fd + fo)

    x = fft.ifft2(Vis)
    x = np.fft.fftshift(x)
    pyfits.writeto('fft_model_'+str(np.round(wave/1e-6, 4))+'_'+year+'.fits', np.absolute(x), overwrite=True)

    x2 = fft.ifft2(Vis2)
    x2 = np.fft.fftshift(x2)
    pyfits.writeto('fft_ring_'+str(np.round(wave/1e-6, 4))+'_'+year+'.fits', np.absolute(x2), overwrite=True)

    return np.absolute(x)
