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
    x0::AbstractVector;
    niter = 10;
    )

    M,N = size(x0)

    x = x0[:];
    z = zeros(size(x));
    η = zeros(size(x));

    for i = 1:niter
        x = update_x(z,η,μ);
        z = update_z(reshape(x+η,M,N))[:]
        η = η + (x-z)
    end

    return x
end
