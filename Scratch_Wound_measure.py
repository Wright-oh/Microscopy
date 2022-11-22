# This code will measure the area of flat surface in a brightfield image - can be used for measuring scratch wound assays.

import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from skimage.filters import threshold_otsu
import cv2

img = cv2.imread(r"**File Path**", 0)
print("img read")
print(img)
print(img.shape)
entropy_img = entropy(img, disk(8))
print("entropy calculated")
thresh = threshold_otsu(entropy_img)
print("threshold applied")
binary = entropy_img <= thresh
print("binary calculated")
scratch_area = np.sum(binary == 1)
print("scratch area calculated")
print("scratch area = " + str(scratch_area))

fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))

img0 = ax0.imshow(img)
ax0.set_title("Original image")
ax0.axis("off")
fig.colorbar(img0, ax=ax0)

img1 = ax1.imshow(entropy_img)
ax1.set_title("Entropy, feature defined")
ax1.axis("off")
fig.colorbar(img1, ax=ax1)

img2 = ax2.imshow(binary)
ax2.set_title("Binary region defined")
ax2.axis("off")
fig.colorbar(img2, ax=ax2)

fig.tight_layout()

plt.show()
