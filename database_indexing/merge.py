import sys, os, shutil

def rename_specs(folder_path):
    print(os.walk(folder_path))

    subdirs = [x[0] for x in os.walk(folder_path)]  
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        for file in files:
            if file.endswith(".vnnlib"): 
                nnet_path = os.path.join(subdir, file)
                new_path = folder_path+"/specs/"+nnet_path.replace("/","_")[2:]
                # print(nnet_path)
                # print(new_path)
                shutil.copy(nnet_path, new_path)
                # return

if __name__ == "__main__":
    rename_specs(sys.argv[1])
