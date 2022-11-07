from patchify import patchify
import tiffile as tiff
from skimage import img_as_float

image = img_as_float(tiff.imread("**File Path**"))
print(image.shape)
patches = patchify(image, (5,5), step=5)
print(patches.shape)
