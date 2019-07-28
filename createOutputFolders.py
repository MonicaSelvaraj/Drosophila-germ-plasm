import os

inputDir = "Stacks"
os.mkdir("Data")
for file in os.listdir(inputDir):
    if file.endswith(".czi"):
        folderName=file[:-4]
        
        #Creating a folder for the current sample
        path = "Data/"+folderName
        os.mkdir(path)
        
        #Creating subfolders within the new folder for ZSlices, ResultsFiles, and Output
        os.mkdir(path+"/ZSlices")
        os.mkdir(path+"/ZResults")
        os.mkdir(path+"/Output")
        
    else:
        continue
