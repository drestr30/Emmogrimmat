# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


def laguerre_gauss_filter(width, height, w=0.9999):

    x = np.arange(-1*(width/2), (width/2), 1)
    y = np.arange(-1*(height/2), (height/2), 1)
    x, y = np.meshgrid(x, y)

    A = (1j*np.square(np.pi)*np.power(w, 4)) # constant term

    dom = (np.square(x)+np.square(y))
    arg = (-np.square(np.pi)) * np.square(w) * dom
    lgcartesian = A*(x + 1j*y)*np.exp(arg)

    return lgcartesian


def fourier_transform(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    return fshift

def inverse_fourier_transform(spectrum):
    imagep = np.fft.ifft2(spectrum)
    imagep = np.fft.ifftshift(imagep)

    return imagep

#a = laguerre_gauss_filter(299,299)
#plt.imshow(np.angle(a), cmap='gray')
#plt.show()
#plt.imshow(np.abs(a), cmap='gray')
#plt.show()

