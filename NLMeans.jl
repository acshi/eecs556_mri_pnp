using Images
using TestImages
using Noise
using PyCall
using Conda

#Conda.add("opencv"; channel = "defaults")

np = pyimport("numpy")
cv = pyimport_conda("cv2", "opencv")

img = testimage("mandril_gray")
img_noise = add_gauss(img, 0.2, 0, clip=true)

pyimg = np.uint8(reinterpret(UInt8, img_noise)) # Convert julia image to np.uint8 array

h = 40

img_denoised = cv.fastNlMeansDenoising(pyimg, h=h)

gray = (reinterpret(Gray{N0f8}, img_denoised)) # convert uint8 array to normed uint8 array 

mosaicview(img_noise, gray; nrow=1)