# load_fastmri_data.jl
# function for loading data from fastmri dataset
# ret3D option controls whether to return 3D volume vs central slice

using HDF5: h5read

function load_fastmri_data(file::String;
    ret3D::Bool = false)

    kspace = h5read(file,"kspace")
    (nkx,nky,nkz) = size(kspace)

    image = ifftq(ifftq(kspace,2),1)

    if ret3D
        return image
    else
        return image[:,:,Int(ceil(nkz/2))]
    end
end
