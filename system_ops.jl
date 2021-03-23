# system_ops.jl
# file defines forward/adjoint MRI system operations
# for use with LinearMapsAA to define false matrices

# inputs/outputs:
# image: MN x 1 image vector
# ksp: MN/R x 1 undersampled k-space vector
# B: M x N sampling mask

# usage: A = LinearMapAA(x -> sys_forw(x,B),x -> sys_adj(x,B), M*N/R, M*N)

function sys_forw(image, B::AbstractMatrix)
    # image -> kspace

    (M,N) = size(B)
    image = reshape(image,(M,N))

    ksp = fftq2(image)
    ksp = ksp[B]

    return ksp
end

function sys_adj(ksp, B::AbstractMatrix)
    # kspace -> image

    (M,N) = size(B)
    ksp_FS = Complex.(zeros(M,N))
    ksp_FS[B] = ksp

    image = ifftq2(ksp_FS)

    return image[:]
end
