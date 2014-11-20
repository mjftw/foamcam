#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <string>
#include <cmath>

#include "img_proc.h"
#include "opdata.h"

void showHelp();

///*TODO* Add Boost Filesystem in order to get file timestamps (cross platform solution)
///*TODO* Make false whitecap detection less frequent (subimg corner type artefacts)

int main(int argc, char *argv[])
{
    string src_path = "E:/FoamCam Project/data/2013-11-02_12-01-24/000004/00004113.img";
    string op_path = "E:/FoamCam Project/output_data";
    bool output_mode = false; //true for simple output, false for advanced output (file structure creation and image output)

    for(int i=1; i<argc; i++)
    {
        if((!strcmp(argv[i], "-s")) && (i+1 <= argc))
            src_path = argv[i+1];
        else if((!strcmp(argv[i], "-d")) && (i+1 <= argc))
            op_path = argv[i+1];
        else if((!strcmp(argv[i], "-m")) && (i+1 <= argc)) ///*TODO* add exception handling for invalid input
            output_mode = argv[i+1];
        else if((!strcmp(argv[i], "-v")) && (i+1 <= argc))
            set_output_mode(atoi(argv[i+1]));
        else if((!strcmp(argv[i], "-c")) && (i+1 <= argc))
        {
            convertRaw(src_path, op_path, argv[i+1]);
            return 0;
        }

        else if(!strcmp(argv[i], "-?"))
        {
            showHelp();
            return 0;
        }
    }

    Mat src = imreadRaw(src_path); ///*TODO* add end program if source not found
    OpData data(src_path, src, output_mode);

    showImg("Source", src);

    //removeBarrelDist(src);

    extractWhitecaps(src, data);

    data.save(op_path);
    waitKey(0);// wait for a keystroke in the window
    return 0;
}


void showHelp()
{
    cout << endl << "Optional command line arguments:" << endl;
    cout << "-s [src] -d [dest] -c [format]" << endl;
    cout << "-s: .img source file" << endl;
    cout << "-d: data output destination directory" << endl;
    cout << "-c: convert .img file to specified format" << endl;
    cout << "-m: output mode, 0 = advanced, 1 = simple (.dat file only, in source directory)" << endl;
    cout << " Overrides other arguments" << endl;
    cout << "-v: view debug images?, 0/1" << endl;
    cout << " Formats: " << SUPPORTED_IMG_FORMATS << endl;
    cout << "-?: help" << endl;
}
