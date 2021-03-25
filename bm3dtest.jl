using Images
using TestImages
using Noise
using BM3D # add "https://github.com/Longhao-Chen/BM3D.jl.git"


img = testimage("mandril_gray")
img_noise = add_gauss(img, 0.2, 0, clip=true)

img_denoised = bm3d(img_noise, 0.2)

assess_psnr(img_noise, img_denoised)