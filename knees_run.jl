using Plots: plot, plot!, savefig
using MIRT: jim, pogm_restart
using Suppressor
using BM3D
using Images: assess_psnr, rmse, imROF
using TestImages
using PyCall
using Noise:add_gauss
using LaTeXStrings
using FileIO:save
using ImageCore:clamp01nan

np = pyimport("numpy")
cv = pyimport_conda("cv2", "opencv")
# Include needed files
include("admm.jl")
include("load_fastmri_data.jl")
include("system_ops.jl")
include("create_sampling_mask.jl")
include("utils.jl")
include("primal.jl")
@suppress include("dncnn.jl") # Suppress weight printing

# Create a single Julia function for calling nlmeans
function nlMeans(img::Array{Float64,2}, h::Number=3)
    pyimg = np.uint8(convert(Array{UInt8,2}, floor.(img / max(img...) * 255))) # Convert julia image to np.uint8 array
    img_denoised = float32.(cv.fastNlMeansDenoising(pyimg, h=h)) ./ 255
end

function nlMeans(img::Array{Float32,2}, h::Number=3)
    pyimg = np.uint8(convert(Array{UInt8,2}, floor.(img / max(img...) * 255))) # Convert julia image to np.uint8 array
    img_denoised = float32.(cv.fastNlMeansDenoising(pyimg, h=h)) ./ 255
end
# Useful to flip images
function rev(img::AbstractArray{T,2}) where T
    img[end:-1:1, end:-1:1]
end

normIm = (x) -> abs.(x) |> (x) -> x / max(x...);

plug_names = ["dncnn", "tv", "nlm", "bm3d", "norm"]
# plug_names = ["nlm"]

tvλ = 0.002
tvIter = 4
nlmH = 3.0
bm3dH = 0.08
# Make a dictionary of the priors
plugs = Dict(
    "dncnn" => (x) -> normIm(dncnn_denoise(float32.(abs.(x)))),
    "tv"    => (x) -> normIm(imROF(abs.(x), tvλ, tvIter)),
    "nlm"   => (x) -> normIm(nlMeans(abs.(x), nlmH)),
    "bm3d"  => (x) -> normIm(bm3d(abs.(x), bm3dH)),
    "norm"  => (x) -> normIm(x),
    );

function float32_to_img(img)
    return reinterpret(Gray{Float32}, map.(clamp01nan, Float32.(img)));
end

function save_results(fName, method, plug_name, R, img, psnr, rmse_vals)
    file_name_stem = fName * "_" * method * "_" * plug_name * "_" * string(R) * "_";
    base_path = joinpath(pwd(), "results")
    save(joinpath(base_path, file_name_stem * "img.png"), float32_to_img(img));
    open(joinpath(base_path, file_name_stem * "psnr.txt"), "w") do file
        write(file, string(psnr) * "\n")
    end
    if length(rmse_vals) > 0
        open(joinpath(base_path, file_name_stem * "rmse.txt"), "w") do file
            write(file, join(rmse_vals, "\n"))
        end
    end
end

function process(fName, R)
    nIter = 10

    # Load a fully sampled image
    fPath = joinpath(pwd(), "data", fName) # Full file path
    # fPath = joinpath(dirname(pwd()), "data", fName) # Full file path
    # Load naturally noisy image
    noisy_img = abs.(load_fastmri_data(fPath))
    # Use a denoised image b/c PnP denoises
    img = normIm(nlMeans(noisy_img, 3))
    # Optionally add more noise
    noisy_img = add_gauss(img, 0.05);
    noisyPSNR = assess_psnr(noisy_img, img) |> (x) -> round(x, digits=2)
    save_results(fName, "", "noisy", R, noisy_img, noisyPSNR, []);

    (M, N) = size(img)
    mask = create_sampling_mask(M, N, R, seed=0)
    save(joinpath(pwd(), "results", string(M) * "_" * string(N) * "_" * string(R) * "_mask.png"), float32_to_img(Float32.(mask)));

    # Determine number of under sampled points
    M_us = Int(round(M / R))
    # Create linear map, A, that describes the MRI system model
    A = LinearMapAA(x -> sys_forw(x, mask), x -> sys_adj(x, mask), (M_us * N, M * N))
    y = A * noisy_img[:]
    # Applying adjoint model to k-space reconstructs the image (poorly)
    adjointIm = abs.(reshape(A' * y, M, N)) |> (x) -> x / max(x...)
    # Make some plots
    psnrAdjoint = assess_psnr(adjointIm, img) # PSNR
    save_results(fName, "", "adjoint", R, adjointIm, psnrAdjoint, []);

    for plug_name in plug_names
        x̂, plug_rmse = @suppress admm(
            (z, η, μ) -> primal(z, η, μ, A, mask, y),
            plugs[plug_name],
            1,
            adjointIm,
            niter=nIter,
            fun=(x, iter) -> rmse(x, img)
        )
        # Reshape and normalize
        x̂ = normIm(reshape(abs.(x̂), M, N))
        # Calculate psnr
        psnr = assess_psnr(x̂, img)

        save_results(fName, "admm", plug_name, R, x̂, psnr, plug_rmse);
    end

    for plug_name in plug_names
        # Reconstruct
        x̂, plug_rmse = pogm_restart(
            adjointIm[:],
            (x) -> undef,
            (x) -> A' * (A * x - y),
            2,
            g_prox=(x, c) -> plugs[plug_name](reshape(x, M, N))[:],
            restart=:none,
            niter=nIter,
            fun=(aa, x, ac, ad) -> rmse(x, img[:])
        )
        # Reshape and normalize
        x̂ = normIm(reshape(abs.(x̂), M, N))
        # Calculate psnr
        psnr = assess_psnr(x̂, img)

        save_results(fName, "pogm", plug_name, R, x̂, psnr, plug_rmse);
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    fName = ARGS[2]

    for R in [4, 8, 12]
        process(fName, R)
    end
end
