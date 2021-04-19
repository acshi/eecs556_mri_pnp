using FileIO

fNames = ["file1000000.h5", "file1000017.h5", "file1000031.h5", "file1000041.h5", "file1000071.h5", "file1000107.h5",
          "file1000114.h5", "file1000153.h5", "file1000182.h5", "file1000196.h5", "file1000007.h5", "file1000026.h5",
          "file1000033.h5", "file1000052.h5", "file1000073.h5", "file1000108.h5", "file1000126.h5", "file1000178.h5",
          "file1000190.h5", "file1000201.h5"]

for R in [4, 8, 12]
    println("R = " * string(R))

    for method in ["admm", "pogm"]
        println("method = " * method)
        for plug_name in ["tv", "dncnn", "bm3d", "nlm"]
            # println("plug = " * plug_name)
            psnr_values = []
            for fName in fNames
                file_name_stem = fName * "_" * method * "_" * plug_name * "_" * string(R) * "_";
                base_path = joinpath(pwd(), "results")
                file_path = joinpath(base_path, file_name_stem * "psnr.txt")
                # println(file_path)
                # save(joinpath(base_path, file_name_stem * "img.png"), float32_to_img(img));
                text = readlines(open(file_path))[1]
                # println(text)
                append!(psnr_values, parse(Float32, text))
            end
            # println(psnr_values)
            mean_psnr = sum(psnr_values) / length(psnr_values)
            print(round(mean_psnr, digits=1))
            print(" & ")
        end
        println()
    end
end
