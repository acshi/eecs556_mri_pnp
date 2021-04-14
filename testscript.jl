# this is the script I've been using to test the different parts/functions of the project

using LinearAlgebra: norm
using LinearMapsAA
using MIRT: jim, pogm_restart
using Plots: plot, plot!

include("utils.jl")
include("load_fastmri_data.jl")
include("create_sampling_mask.jl")
include("system_ops.jl")
include("admm.jl")
include("primal.jl")
include("dncnn.jl")

# load in data from fastmri file
imref = load_fastmri_data("file1000000.h5")
(M,N) = size(imref);
j1 = jim(abs.(imref[end:-1:1,end:-1:1]),title="reference")

# def acceleration factor for sampling mask
R = 5
Mus = Int(round(M/R))

# create 1D random undersampling mask
B = create_sampling_mask(M,N,R,seed=0)

# create linear map A that describes the MRI system model
A = LinearMapAA(x -> sys_forw(x,B),x -> sys_adj(x,B), (Mus*N, M*N))

# applying model to reference image generates an undersampled k-space
y = A*imref[:]

# applying adjoint model to k-space reconstructs the image (poorly)
x0 = reshape(A'*y,M,N)
j2 = jim(abs.(x0[end:-1:1,end:-1:1]),title="zero-filled recon")

nrmse = (x) -> norm(normalize(x).-normalize(imref[:]),2)

# recon data w/ admm algorithm
# the primal function is still a little unstable. If you're testing another
# denoiser, lmk, as I'd like to double check it once we have those ready
x_admm, nrmse_admm = admm((z,η,μ) -> primal(z,η,μ,A,B,y), (x) -> dncnn_denoise(Float32.(abs.(x))), 1, x0, niter=10, fun=(x,i)->nrmse(x))
x_admm = reshape(x_admm,M,N)

j3 = jim(abs.(x_admm[end:-1:1,end:-1:1]),title="admm recon")
j4 = jim(abs.(imref[end:-1:1,end:-1:1]) - abs.(x_admm[end:-1:1,end:-1:1]),title="admm sub")

x_pogm, nrmse_pogm = pogm_restart(x0[:], (x) -> undef, (x) -> A'*(A*x-y), 2,
    g_prox=(x,c) -> dncnn_denoise(Float32.(abs.(reshape(x,M,N))))[:],
    restart=:none,niter=10,fun=(aa,x,ac,ad) -> nrmse(x))

x_pogm = reshape(x_pogm,M,N)

j5 = jim(x_pogm[end:-1:1,end:-1:1],title="pogm recon")
j6 = jim(abs.(imref[end:-1:1,end:-1:1]) - abs.(x_pogm[end:-1:1,end:-1:1]),title="pogm sub")

plot(j1,j2,j3,j4,j5,j6)

plot(0:10,nrmse_admm)
plot!(0:10,nrmse_pogm)
