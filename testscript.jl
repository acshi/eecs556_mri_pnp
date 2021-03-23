# this is the script I've been using to test the different parts/functions of the project

using LinearMapsAA
using MIRT: jim
using Plots: plot

include("utils.jl")
include("load_fastmri_data.jl")
include("create_sampling_mask.jl")
include("system_ops.jl")
include("admm.jl")
include("primal.jl")

# load in data from fastmri file
imref = load_fastmri_data("file1000000.h5")
(M,N) = size(imref);
j1 = jim(abs.(imref[end:-1:1,end:-1:1]),title="reference")

# def acceleration factor for sampling mask
R = 3.76

# create 1D random undersampling mask
B = create_sampling_mask(M,N,R,seed=0)
R = M*N/sum(B)

# create linear map/false matrix A that describes the MRI system model
A = LinearMapAA(x -> sys_forw(x,B),x -> sys_adj(x,B), (Int(M*N/R), M*N))

# applying model to reference image generates an undersampled k-space
y = A*imref[:]
j2 = jim(log.(abs.(reshape(y,Int(round(M/R)),N))),title="k-space")

# applying adjoint model to k-space reconstructs the image (poorly)
x0 = reshape(A'*y,M,N)
j3 = jim(abs.(x0[end:-1:1,end:-1:1]),title="recon")

j4 = jim(abs.(imref[end:-1:1,end:-1:1]) - abs.(x0[end:-1:1,end:-1:1]),title="subtraction")

plot(j1,j2,j3,j4)


# recon data w/ admm algorithm
# to-do: write/load denoiser function
x_hat = admm((z,η,μ) -> primal(z,η,μ,A,B,y), denoiser, 1, x0, niter=10)
