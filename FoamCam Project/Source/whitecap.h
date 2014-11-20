#ifndef WHITECAP_H
#define WHITECAP_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class Whitecap
{
    public:
        Whitecap(vector<Point> new_contour);
        virtual ~Whitecap();

        Mat getMask();
        vector<Point> getContour();
        Point getCentroid();
        int getSize();
        int getIntensity();

        void setContour(vector<Point> new_contour);

    private:
        vector<Point>* contour;
        Point* centroid;
        Mat* mask;
        int intensity;
        int total_size;

        void calcCentroid();
        void calcMask();
        void calcIntensity();
        void calcSize();

};

#endif // WHITECAP_H
