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

function primal(z::AbstractVector,
    η::AbstractVector,
    μ::Real,
    A::Any,
    B::AbstractMatrix,
    y::AbstractVector)

    (M,N) = size(B)

    xgrad = reshape(A'*y .+ μ*(z-η), (M,N))
    binv = 1 ./(B.+μ)

    x = ifftq2(binv.*fftq2(xgrad))

    return x[:]

end
