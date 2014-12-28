#ifndef OPDATA_H
#define OPDATA_H

#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define DEFAULT_OP_IMG_EXT ".jpg"
///*TODO* Use the open source library Boost Filesystem in order to make directory creation cross platform.
using namespace std;
using namespace cv;

class OpData
{
    public:
        OpData(string new_src_img_path, bool simple_save = false);
        OpData(string new_src_img_path, Mat& new_src_img, bool simple_save = false);
        ~OpData();
        void addImg(string name, Mat& img);
        void addField(string field, string value);
        void addField(string field, int value);
        void addField(string field, float value);

        bool save(string dest_dir = "", string format = DEFAULT_OP_IMG_EXT);
        bool saveSimple(string filename = "default_output.csv");
        bool saveImg(string dest_dir = "", string format = DEFAULT_OP_IMG_EXT);

        string getImgName();
        string getImgDir();

    private:

        vector<pair<string, string>*>* fields;
        vector<pair<string, Mat*>*>* imgs;
        Mat* src_img;
        string* src_img_path;

        bool simple_output; //saves only the .dat file in the source image directory, does not save data images.

        bool makeDir(string path);  //different function used for unix and windows
        string getTimestamp();
};

#endif // OPDATA_H
