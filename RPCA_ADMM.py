import sys, os
import numpy as np
import glob
from matplotlib import pylab as plt
from numpy.linalg import svd
from PIL import Image
import time

class TRPCA:

    def converged(self, L, E, X, L_new, E_new):
        eps = 1e-8
        condition1 = np.max(L_new - L) < eps
        condition2 = np.max(E_new - E) < eps
        condition3 = np.max(L_new + E_new - X) < eps
        return condition1 and condition2 and condition3

    def SoftShrink(self, X, tau):
        z = np.sign(X) * (abs(X) - tau) * (np.sign(abs(X) - tau) + 1)/ 2
        return z

    def SVDShrink(self, X, tau):
        u, s, v = svd(X, full_matrices = False)
        s_bar = self.SoftShrink(s, tau)
        return np.dot(np.dot(u, np.diag(s_bar)), v)


    def ADMM(self, X):
        m, n = X.shape
        rho = 1.5
        mu = 1e-3
        mu_max = 1e10
        max_iters = 1000
        lamb = 1/np.sqrt(max(m, n))
        L = np.zeros((m, n), float)
        E = np.zeros((m, n), float)
        Y = np.zeros((m, n), float)
        iters = 0
        while True:
            iters += 1
            L_new = self.SVDShrink(X - E - (1/mu) * Y, 1/mu)
            E_new = self.SoftShrink(X - L_new - (1/mu) * Y, lamb/mu)
            Y += mu * (L_new + E_new - X)
            mu = min(rho * mu, mu_max)
            if self.converged(L, E, X, L_new, E_new) or iters >= max_iters:
                return L_new, E_new
            else:
                L, E = L_new, E_new
                print(np.max(X - L - E))


# for image denoising
# Load input data as gray image
# set the file path of your photo which you want to denoise
X = np.array(Image.open('set your path').convert('L'))
TRPCA = TRPCA()
L, S = TRPCA.ADMM(X)
plt.subplot(131)
plt.imshow(L)
plt.gray()
plt.subplot(132)
plt.imshow(S)
plt.gray()
plt.subplot(133)
plt.imshow(X)
plt.gray()
plt.show()

# for background monitoring
# files = glob.glob('set your directory/*.jpg')
# matrix = []
# for file in files:
#    img = Image.open(file).convert('L')
#    h,w=np.array(img).shape
#    pixels = list(img.getdata())
#    matrix.append(pixels)
# X=(np.array(matrix).astype(np.float64))
# TRPCA = TRPCA()
# L, S = TRPCA.ADMM(X)
# L = L[:,0].reshape((h,w))
# S = S[:,0].reshape((h,w))
