# admm optimization algorithm

# inputs:
# update_x: function for x update, in terms of z, η, and μ
# update_z: image denoiser function
# μ: η update scaling constant
# x0: initial guess for x
#
# options:
# niter: # of ADMM iterations
#
# outputs:
# x: final estimate for x

function admm(update_x::Function,
    update_z::Function,
    μ::Real,
    x0::AbstractMatrix;
    niter = 10)

    (M,N) = size(x0)

    z = x0[:]
    η = Complex.(zeros(size(z)));
    x = update_x(z,η,μ)

    for i = 1:niter
        display(i)

        z = update_z(reshape(x.+η,M,N))[:]
        η = η .+ (x.-z)

        x = update_x(z,η,μ);
    end

    return x
end
