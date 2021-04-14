# admm optimization algorithm

# inputs:
# update_x: function for x update, in terms of z, η, and μ
# update_z: image denoiser function
# μ: η update scaling constant
# x0: initial guess for x
#
# options:
# niter: # of ADMM iterations
# fun: monitoring function, evaluated at each iter (eg, (x,i) -> rmse(x))
#
# outputs:
# x: final estimate for x
# funout: iter+1 vector of monitoring function outputs

function admm(update_x::Function,
    update_z::Function,
    μ::Real,
    x0::AbstractMatrix;
    niter=10,
    fun::Function=(x,iter) -> 0)

    (M,N) = size(x0)
    funout = zeros(niter+1)

    z = x0[:]
    η = Complex.(zeros(size(z)));
    x = update_x(z,η,μ)
    funout[1] = fun(x,0)

    for i = 1:niter
        @info "iter $i"

        z = update_z(reshape(x.+η,M,N))[:]
        η = η .+ (x.-z)

        x = update_x(z,η,μ);
        funout[i+1] = fun(x,i)
    end

    if sum(funout .!= 0) > 0
        return x, funout
    else
        return x
    end
end
