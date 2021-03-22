# create_sampling_mask.jl
# creates a sampling mask based on 1D undersampling pattern described by
# Lustig, et al. (MRM 2007)
# note that this generates along the first (PE) dimension- if you want to play
# with RO-undersampling, swap M/N inputs and use B'.

# to-do: doesn't currently have any control for generating masks with a specific acceleration factor

using Random: seed!

function create_sampling_mask(M::Int,N::Int,R;
    seed::Int=0)

    # set seed
    seed!(0);

    # generate 1D quadratic undersampling pattern
    kpe = 2*(-M/2:1:M/2-1)/M
    qfun = x -> 0.75*(abs(x)-1)^2
    b = rand(M) .< qfun.(kpe)

    return Bool.(b*ones(N)')

end
