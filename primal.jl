# primal update function for admm implementation
#
# ex: admm((z,η,μ) -> primal(z,η,μ,A,B,y), ...)
#
# inputs:
# z, η, μ: inputs from admm function. anonymize wrt these
# A: MRI system matrix/lin map
# B: sampling mask
# y: original signal vector
#
# outputs:
# x: updated value for x.

using LinearMapsAA

function primal(z::Any,
    η::Any,
    μ::Real,
    A::Any,
    B::Any,
    y::Any)

    (M,N) = size(B)

    # to-do: the derivation here is definitely worth spot-checking w/ Fessler
    xgrad = reshape(A'*y + μ*(z-η), (M,N))
    binv = 1 ./(B.+μ)

    x = fftq2(binv.*ifftq2(xgrad))

    return x[:]

end
