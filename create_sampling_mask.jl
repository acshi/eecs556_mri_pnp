# create_sampling_mask.jl
# creates a sampling mask based on 1D undersampling pattern described by
# Lustig, et al. (MRM 2007)
# note that this generates along the first (PE) dimension- if you want to play
# with RO-undersampling, swap M/N inputs and use B'.
#
# inputs:
# M,N: matrix size
# R: acceleration factor
# seed: seed for RNG, for random mask values
#
# outputs:
# B: boolean sampling mask

using Random: seed!

function create_sampling_mask(M::Int,N::Int,R;
    seed::Int=0)

    # set seed
    seed!(seed);
    rm=rand(M)

    # generate 1D polynomial undersampling pattern
    kpe = 2*(-M/2:1:M/2-1)/M
    qfun = (x,p) -> abs(abs(x)-1)^p

    # run a bin search on p to create a mask w/ specific acceleration factor
    b = 0;
    pmin = 1; pmax = 10;

    while sum(b) != round(M/R)
        pc = (pmin+pmax)/2
        b = rm .< qfun.(kpe,pc)

        if sum(b) > round(M/R)
            pmin = pc
        else
            pmax = pc
        end
    end

    return Bool.(b*ones(N)')

end
