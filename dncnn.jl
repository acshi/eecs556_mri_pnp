# DnCNN

# def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
# m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
# m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
# m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

# self.model = B.sequential(m_head, *m_body, m_tail)

# head with mode CR
# body with modes CBR
# tail with mode C

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
# def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
#     L = []
#     for t in mode:
#         if t == 'C':
#             L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
#         elif t == 'B':
#             L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
#         elif t == 'R':
#             L.append(nn.ReLU(inplace=True))
using Flux
using JSON
using Images
using ImageCore:clamp01nan
using ImageFiltering
using Noise

function create_dncnn()
    model_json = JSON.parsefile(joinpath(dirname(@__FILE__), "dncnn_model.json"));

    filter_size = (3, 3);
    # Conv layers take weight and bias parameters to set them

    weight = Float32.(reshape(model_json["model.0.weight"][2], (3, 3, 1, 64)));
    bias = Float32.(model_json["model.0.bias"][2]);
    head = Conv(filter_size, 1 => 64, relu, pad=1, weight=weight, bias=bias);

    body_layers = [];

    for i in 1:18
        weight = Float32.(reshape(model_json["model." * string(i * 2) * ".weight"][2], (3, 3, 64, 64)));
        bias = Float32.(model_json["model." * string(i * 2) * ".bias"][2]);
        body_layer = Conv(filter_size, 64 => 64, relu, pad=1, weight=weight, bias=bias);
        body_layers = vcat(body_layers, body_layer);
    end

    # body = [BatchNorm(64, relu, momentum=0.9, ϵ=1e-4) ∘ Conv(filter_size, 64 => 64) for _ in 1:18];
    # body = [Conv(filter_size, 64 => 64, relu) for _ in 1:18];

    weight = Float32.(reshape(model_json["model.38.weight"][2], (3, 3, 64, 1)));
    bias = Float32.(model_json["model.38.bias"][2]);
    tail = Conv(filter_size, 64 => 1, pad=1, weight=weight, bias=bias);

    layers = vcat(head, body_layers, tail);

    for layer in layers
        println([size(k) for (k, v) in params(layer).params.dict]);
    end

    model(x) = foldl((x, m) -> m(x), layers, init=x);
    return model;
end

function padto(img, dims)
    padded = zeros(Float32, dims);

    offset1 = Int((dims[1] - size(img)[1]) / 2);
    offset2 = Int((dims[2] - size(img)[2]) / 2);

    padded[1 + offset1:offset1 + size(img)[1], 1 + offset2:offset2 + size(img)[2]] = img;

    return padded;
end

net = create_dncnn();

function dncnn_denoise(img)
    denoising_residual = net(Flux.batch([Flux.batch([img])]));

    # padded_residual = padto(denoising_residual, size(img));
    padded_residual = denoising_residual;
    denoised_img = clamp01nan.(img .- padded_residual);

    return denoised_img[:,:];
end

function img_to_float32(img)
    return reinterpret(Float32, float.(Gray.(img)));
end

function float32_to_img(img)
    return reinterpret(Gray{Float32}, img);
end

if abspath(PROGRAM_FILE) == @__FILE__
    using FileIO:load,save

    img = img_to_float32(load(joinpath(dirname(@__FILE__), "data/cameraman.png")));
    # img = img_to_float32(load("eecs556_mri_pnp/data/cameraman.png"));

    σNoise = 0.2
    μNoise = 0.0
    noisy_img = clamp01nan.(add_gauss(img, σNoise, μNoise, clip=true));

    denoised_img = dncnn_denoise(noisy_img);

    noisy_psnr = assess_psnr(noisy_img, img);
    println("Noisy PSNR: " * string(noisy_psnr));

    dncnn_psnr = assess_psnr(denoised_img, img);
    println("DnCNN PSNR: " * string(dncnn_psnr));

    # save("eecs556_mri_pnp/results/cameraman.png", float32_to_img(img));
    # save("eecs556_mri_pnp/results/noisy_cameraman.png", float32_to_img(noisy_img));
    # save("eecs556_mri_pnp/results/denoised_cameraman.png", float32_to_img(denoised_img));

    save(joinpath(dirname(@__FILE__), "results/noisy_cameraman.png"), float32_to_img(noisy_img));
    save(joinpath(dirname(@__FILE__), "results/denoised_cameraman.png"), float32_to_img(denoised_img));
end

# W = rand(2, 5)
# b = rand(2)

# predict(x) = W * x .+ b

# function loss(x, y)
#     ŷ = predict(x)
#     sum((y .- ŷ).^2)
# end

# x, y = rand(5), rand(2) # Dummy data
# loss(x, y) # ~ 3

# gs = gradient(() -> loss(x, y), params(W, b))
# W .-= 0.1 .* gs[W]
