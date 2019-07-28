/* 
DrosophilaGP Macro
*/

//Storing the path to the input files 
inputDir ="/Users/ms12122/Desktop/DrosophilaGP/Stacks/";

/* 
Sometimes, ImageJ can get confused when it has to open or close windows and perform operations on them, in which case it can appear as if operations are called out of order. To prevent that, enable the batch mode
(https://imagej.net/How_to_apply_a_common_operation_to_a_complete_directory)
*/
setBatchMode(true);

//Reading the files in the input directory 
list = getFileList(inputDir);

//MAIN LOOP
for(i=0;i<list.length;i++){
//The following actions are performed for each z-stack

    open(inputDir+list[i]); //TESTED
    run("Duplicate...", "duplicate"); //TESTED
	run("Auto Threshold", "method=Yen ignore_black white use_stack_histogram"); //TESTED
	outputFilename = substring(list[i],0,lengthOf(list[i])-4);
	thresholdingOutputPath = "/Users/ms12122/Desktop/DrosophilaGP/Data/"+outputFilename+"/ZSlices/";
    run("Image Sequence... ", "format=TIFF save=thresholdingOutputPath");
    resultsOutputPath = "/Users/ms12122/Desktop/DrosophilaGP/Data/"+outputFilename+"/ZResults/";
    zSliceList = getFileList(thresholdingOutputPath);
    for(j=0;j<zSliceList.length;j++){
        open(thresholdingOutputPath+zSliceList[j]);
        run("Image to Results"); //Results files are saved as Results0, Results1, ..
        resultsFileName = "Results"+j;
        saveAs("Results",resultsOutputPath+resultsFileName + ".csv");
        close();
    }close("*");
    selectWindow("Results"); 
    run("Close");
}
setBatchMode(false);
