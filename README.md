 ### knees_run_all.jl
 Runs our method with ADMM and POGM using each of the main plugs, Non-local means, BM3D, TV, DnCNN, on each of 20 images (not included because the 20 image data files total more than 1GB). This runs everything in serial and so could take many hours to complete. It stores all the results as individual files under the results directory.
 
 ### knees_psnr_results.jl
 Computes overall PSNR values from the results given by knees_run_all.jl
