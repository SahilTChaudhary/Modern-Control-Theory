import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('CMU_Grayscale.png', cv2.IMREAD_GRAYSCALE)
U, Sigma, Vt = np.linalg.svd(img, full_matrices=False)

def compression(cr):
    n = int(1 / (cr))
    k = int((img.shape[0] * img.shape[1]) / (n * (1 + img.shape[0] + img.shape[1])))
    Uk = U[:, :k]
    Sigmak = np.diag(Sigma[:k])
    Vt_k = Vt[:k, :]
    compressed_image = np.dot(np.dot(Uk, Sigmak), Vt_k)
    storage= (Uk.size + Sigmak.size + Vt_k.size)
    print(f'Number of singular values for compression level {cr} are {k}, and storage used is {storage}')
    return compressed_image, k

# Original image
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

# 50% 
compressed_image,n = compression(0.5)
plt.subplot(2,2,2)
plt.imshow(compressed_image, cmap='gray')
plt.title(f'Compression Level: 50% and {n} Singular Values')

# 10%
compressed_image,n = compression(0.1)
plt.subplot(2,2,3)
plt.imshow(compressed_image, cmap='gray')
plt.title(f'Compression Level: 90% and {n} Singular Values')

# 5%
compressed_image,n = compression(0.05)
plt.subplot(2,2,4)
plt.imshow(compressed_image, cmap='gray')
plt.title(f'Compression Level: 95% and {n} Singular Values')

plt.show()
