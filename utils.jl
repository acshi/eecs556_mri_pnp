# utils.jl
# this is just a script I created to throw all our miscellaneous functions in
# so they're not crowding out the main script

using FFTW: fft, ifft, fftshift, ifftshift

# NYU fft functions- they use shifts on both sides of the transform
fftq = (k,d) -> ifftshift(fft(fftshift(k./size(k,d),d),d),d)
ifftq = (k,d) -> fftshift(ifft(ifftshift(k,d),d),d).*size(k,d)

# also want some 2D versions
fftq2 = (k) -> fftq(fftq(k,2),1)
ifftq2 = (k) -> ifftq(ifftq(k,2),1)
