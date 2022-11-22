# This code will cut image Raw_image into several smaller output images. Adjust file names as required

from patchify import patchify
from skimage import img_as_ubyte
import os
import tifffile as tiff
import cv2
# Adjust this code to patch individual images.

# Open image and cut it into smaller images for processing
Raw_image = 'LOECs/IMAGE_1.tif'
img_large = img_as_ubyte(cv2.imread(Raw_image, 1))
filename, file_extension = os.path.splitext(Raw_image)
print('filename is ' + filename)
print('file ext is ' + file_extension)

img_patch = patchify(img_large, (256, 256, 3), step=256)  # Step=256 for 256 patches means no overlap

for i in range(img_patch.shape[0]):
    for j in range(img_patch.shape[1]):
        single_patch_img = img_patch[i, j, :, :]
        tiff.imwrite('LOECs/Output_image/' + 'Output_image_1_' + str(i) + '_' + str(j)+".tif", single_patch_img)
