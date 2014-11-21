#include "opdata.h"

using namespace std;
using namespace cv;

#define WHITE 255
#define BLACK 0

#define NROWS 2048
#define NCOLS 2048

#define SUPPORTED_IMG_FORMATS ".bmp .dib .jpeg .jpg .jpe .jp2 .png .pbm .pgm .ppm .sr .ras .tiff .tif"
#define CAM_PARAM_XML "D:/Libaries/Uni Work/Year 3/foamcam_project/FoamCam Project/calibration/camera_calibration/calibration_data.xml"

#define BORDER_CENT Point(1035,1115) //ellipse parameters found via trial and error
#define BORDER_X_RADIUS 835
#define BORDER_Y_RADIUS 775

Mat imreadRaw(string src_path);
void convertRaw(string src_path, string op_path, string format);

void maskObjects(Mat& src);
void maskBorder(Mat& src);
void maskFrame(Mat& src);
void maskRope(Mat& src);
void extractWhitecaps(Mat& src, OpData& data);
int removeBarrelDist(Mat& src);
void findSkeleton(Mat& src, Mat& dst); // ------------UNUSED------------
void centreImage(Mat& src);

void divideIntoSubimgs(Mat& src, vector<Mat>* op, int n, int m);
void combineSubimgs(vector<Mat>* src, Mat& op, int n, int m);
void optimalThreshSubimgs(Mat& src, Mat& dst, int min_fg_bg_diff, int n, int m);
void getHist(Mat& src, Mat& dst, Mat& mask);
void showHist(Mat& hist);
float getHistVariance(Mat& hist); // ------------UNUSED------------
int getHistPeak(Mat& hist);

Mat overlayContours(Mat& src, vector<vector<Point> >* contours);

void path(Mat& img, Point* pts, int npts, const Scalar& color, int thickness, int lineType, int shift);
void labelPolyPoints(Mat& img, Point* pts, int npts, const Scalar& color, double text_scale);
void showMaskOverlay(Mat& src, Mat& mask);
void findContourCentroids(vector<vector<Point> >& contours, vector<Point>& output_array);
bool testIfProcessable(Mat& img);
void extractContourData(vector<vector<Point> >& contours, OpData& data);
void showImg(string window_name, Mat& img);

bool checkContourCorners(vector<Point>* contour);


static bool SHOW_DEBUG_IMGS = true; //defines whether image windows should be created to show debug images
///*TODO* Change program defaults to perhaps load from a file

// ---------------- POLYS ----------------

Point FRAME_POLY[] =
{//Points specified explicitly for speed
    Point(1069, 1900),
    Point(1102, 1882),
    Point(1098, 1852),
    Point(1095, 1823),
    Point(1137, 1812),
    Point(1170, 1799),
    Point(1195, 1777),
    Point(1205, 1750),
    Point(1202, 1735),
    Point(1165, 1674),
    Point(1166, 1656),
    Point(1163, 1640),
    Point(1153, 1624),
    Point(1143, 1606),
    Point(1131, 1586),
    Point(1121, 1577),
    Point(1114, 1571),
    Point(1104, 1546),
    Point(1094, 1520),
    Point(1122, 1516),
    Point(1163, 1511),
    Point(1200, 1505),
    Point(1210, 1504),
    Point(1234, 1519),
    Point(1251, 1523),
    Point(1287, 1506),
    Point(1309, 1490),
    Point(1345, 1487),
    Point(1389, 1481),
    Point(1430, 1475),
    Point(1485, 1464),
    Point(1548, 1447),
    Point(1604, 1427),
    Point(1653, 1407),
    Point(1697, 1384),
    Point(1748, 1355),
    Point(1802, 1316),
    Point(1838, 1281),
    Point(1865, 1250),
    Point(1876, 1239),
    Point(1883, 1209),
    Point(1892, 1164),
    Point(1901, 1120),
    Point(1881, 1127),
    Point(1856, 1165),
    Point(1822, 1206),
    Point(1783, 1243),
    Point(1722, 1293),
    Point(1665, 1329),
    Point(1596, 1366),
    Point(1519, 1397),
    Point(1420, 1428),
    Point(1391, 1436),
    Point(1360, 1441),
    Point(1359, 1438),
    Point(1355, 1433),
    Point(1347, 1433),
    Point(1338, 1438),
    Point(1324, 1422),
    Point(1316, 1417),
    Point(1304, 1414),
    Point(1294, 1415),
    Point(1282, 1421),
    Point(1271, 1430),
    Point(1255, 1443),
    Point(1243, 1450),
    Point(1231, 1450),
    Point(1228, 1447),
    Point(1210, 1450),
    Point(1178, 1454),
    Point(1140, 1457),
    Point(1120, 1458),
    Point(1119, 1450),
    Point(1113, 1454),
    Point(1104, 1457),
    Point(1088, 1457),
    Point(1077, 1457),
    Point(1064, 1346),
    Point(1051, 1217),
    Point(1137, 1213),
    Point(1223, 1202),
    Point(1223, 1194),
    Point(1213, 1165),
    Point(1230, 1147),
    Point(1228, 1139),
    Point(1127, 1145),
    Point(1043, 1148),
    Point(1035, 1081),
    Point(1029, 1021),
    Point(997, 1021),
    Point(995, 997),
    Point(988, 998),
    Point(986, 1001),
    Point(986, 1076),
    Point(989, 1143),
    Point(994, 1237),
    Point(1000, 1328),
    Point(1011, 1454),
    Point(973, 1453),
    Point(922, 1454),
    Point(896, 1453),
    Point(844, 1446),
    Point(806, 1438),
    Point(792, 1437),
    Point(785, 1430),
    Point(778, 1430),
    Point(774, 1418),
    Point(763, 1414),
    Point(755, 1410),
    Point(741, 1407),
    Point(725, 1409),
    Point(718, 1416),
    Point(711, 1424),
    Point(701, 1424),
    Point(689, 1426),
    Point(684, 1431),
    Point(629, 1418),
    Point(577, 1403),
    Point(514, 1380),
    Point(456, 1353),
    Point(402, 1325),
    Point(343, 1286),
    Point(253, 1208),
    Point(208, 1156),
    Point(171, 1107),
    Point(168, 1169),
    Point(177, 1224),
    Point(193, 1251),
    Point(239, 1300),
    Point(304, 1364),
    Point(396, 1418),
    Point(446, 1428),
    Point(471, 1421),
    Point(559, 1451),
    Point(649, 1474),
    Point(752, 1494),
    Point(787, 1508),
    Point(811, 1503),
    Point(840, 1499),
    Point(882, 1508),
    Point(950, 1516),
    Point(981, 1517),
    Point(982, 1559),
    Point(975, 1582),
    Point(963, 1595),
    Point(946, 1656),
    Point(944, 1674),
    Point(946, 1685),
    Point(952, 1693),
    Point(941, 1729),
    Point(935, 1756),
    Point(934, 1776),
    Point(945, 1792),
    Point(958, 1803),
    Point(958, 1826),
    Point(972, 1838),
    Point(996, 1844),
    Point(1034, 1844),
    Point(1034, 1885),
    Point(1069, 1900)
};

Point ROPE_POLY[] =
{
    Point(358, 622),
    Point(300, 721),
    Point(548, 987),
    Point(777, 1066),
    Point(780, 999),
    Point(760, 983),
    Point(770, 954),
    Point(864, 975),
    Point(868, 953),
    Point(865, 924),
    Point(675, 818)
};

struct contourSort
{
    inline bool operator() (const vector<Point> contour1, const vector<Point> contour2)
    {
        return (contourArea(contour1, false) < contourArea(contour2, false));
    }
};

Mat imreadRaw(string src_path)
{//Reads a .img file and converts it to a Mat object
    Mat image(NROWS, NCOLS, CV_8UC1, Scalar::all(WHITE));

    unsigned char grayl;

    fstream src_file;
    src_file.open(src_path.c_str(), ios::in | ios::binary);
    if(!src_file.is_open())
    {
        cerr << "Could not open source file." << endl;
        Mat err;
        return err;
    }

    int i,j;
    uchar* p;
    for(i=0; i<NROWS; ++i)
    {
        p = image.ptr<uchar>(i);
        for(j=NCOLS; j>0; --j)
        {
            src_file >> noskipws >> grayl;
            p[j] = grayl;
        }
    }

    Point2f src_center((float)NCOLS/2, (float)NROWS/2);
    Mat rot_mat = getRotationMatrix2D(src_center, 90, 1);
    warpAffine(image, image, rot_mat, Size(NCOLS, NROWS));

    return image;
}

void set_output_mode(bool new_mode)
{
    SHOW_DEBUG_IMGS = new_mode;
    return;
}

void showImg(string window_name, Mat& img)
{
    if(SHOW_DEBUG_IMGS)
    {
    namedWindow(window_name, CV_WINDOW_NORMAL);
    imshow(window_name, img);
    }

    return;
}

bool testIfProcessable(Mat& img)
{///*TODO* must correct fisheye distortion in order to work on this section properly
    bool Proc = true;

// -------- FIND HORIZON ----------
/*

    medianBlur(img, mask, 15);

    showImg("blur", mask);

    Laplacian(mask, mask, CV_8U, 5, 1, 0, BORDER_DEFAULT); ///*TODO* Change to first order function to reduce noise

    mask &= border_mask;
    mask &= frame_mask;
    mask &= rope_mask;

    showImg("laplacian", mask);
// ------------UNUSED------------
    threshold(mask, mask, 40, 255, THRESH_BINARY);

    showImg("thresh1", mask);


    //removes noise
    GaussianBlur(mask, mask, Size(5,5), 1, 1, BORDER_DEFAULT);
    threshold(mask, mask, 0, 255, THRESH_BINARY | CV_THRESH_OTSU);
    GaussianBlur(mask, mask, Size(5,5), 1, 1, BORDER_DEFAULT);
    threshold(mask, mask, 0, 255, THRESH_BINARY | CV_THRESH_OTSU);

    showImg("thresh2", mask);


    int morph_size = 6;
    Mat element = getStructuringElement(2, Size(2*morph_size+1, 2*morph_size+1), Point(morph_size, morph_size));
    morphologyEx(mask, mask, MORPH_CLOSE, element);

    showImg( "morph", mask);

*/


    return Proc;
}

void path(Mat& img, Point* pts, int npts, const Scalar& color, int thickness, int line_type, int shift)
{//Debug tool. Draws a path between the given points.
    int i;
    for(i=0; i<(npts-1); i++)
    {
        line(img, pts[i], pts[i+1], color, thickness, line_type, shift);
    }
    return;
}

void labelPolyPoints(Mat& img, Point* pts, int npts, const Scalar& color, double text_scale)
{//Debug tool. Labels all the points in the poly with a circle and their index number
    int i;
    char buffer[8];
    for(i=0; i<npts; i++)
    {
        circle(img, pts[i], 5, color, 2, 8, 0);
        itoa(i, buffer, 10);
        putText(img, buffer, Point(pts[i].x, pts[i].y + 10), FONT_HERSHEY_SIMPLEX, text_scale, color, text_scale*2, 8, false);
    }
    return;
}

int removeBarrelDist(Mat& src)
{

    Mat cameraMatrix, distCoeffs, map1, map2;

    FileStorage fs;
    fs.open(CAM_PARAM_XML, FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "Failed to open distortion coefficients xml file" << endl;
        return -1;
    }

    fs["Camera_Matrix"] >> cameraMatrix;
    fs["Distortion_Coefficients"] >> distCoeffs;

    centreImage(src);
    //showImg("Source - Centred", src);

    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, src.size(), 1, src.size(), 0),
            src.size(), CV_16SC2, map1, map2);
    remap(src, src, map1, map2, INTER_LINEAR);

    Mat temp1;
    float z = 7; //zoom factor
    resize(src, temp1, Point(0,0), z, z);
    int offset = z*50;
    Mat temp2(temp1(Rect(Point(temp1.rows*(0.5 - 1/z) - offset, temp1.cols*(0.5 - 1/z) - offset), Point(temp1.rows*(0.5 + 1/z) - offset, temp1.cols*(0.5 + 1/z) - offset))));
    resize(temp2, src, Size(0,0), 0.5, 0.5);

    return 0;
}

void centreImage(Mat& src)
{
    Point roiPt1 = Point(BORDER_CENT.x - BORDER_X_RADIUS, BORDER_CENT.y - BORDER_Y_RADIUS);
    Point roiPt2 = Point(BORDER_CENT.x + BORDER_X_RADIUS, BORDER_CENT.y + BORDER_Y_RADIUS);
    Point outPt1 = Point(NROWS/2 - BORDER_X_RADIUS, NCOLS/2 - BORDER_Y_RADIUS);
    Point outPt2 = Point(NROWS/2 + BORDER_X_RADIUS, NCOLS/2 + BORDER_Y_RADIUS);

    Mat temp1(src(Rect(roiPt1, roiPt2)));

    Mat temp2(NROWS, NCOLS, CV_8UC1, Scalar::all(BLACK));

    temp1.copyTo(temp2(Rect(outPt1, outPt2)));
    src = temp2.clone();

    return;
}

void maskObjects(Mat& src)
{
    maskBorder(src);
    maskFrame(src);
    maskRope(src);
    return;
}
void maskBorder(Mat& src)
{
    Mat border_mask(NROWS, NCOLS, CV_8UC1, Scalar::all(BLACK));
    vector<Point> ellipse_poly_vec;
    ellipse2Poly(BORDER_CENT, Size(BORDER_X_RADIUS, BORDER_Y_RADIUS), 0, 0, 360, 1, ellipse_poly_vec);
    Point*  mask_poly = &ellipse_poly_vec[0];
    fillConvexPoly(border_mask, mask_poly, 360, Scalar::all(WHITE), 8, 0);
    src &= border_mask;
    return;
}

void maskFrame(Mat& src)
{
    Mat frame_mask(NROWS, NCOLS, CV_8UC1, Scalar::all(WHITE));
    vector<Point> frame_poly_vec(FRAME_POLY, FRAME_POLY + sizeof(FRAME_POLY) / sizeof(FRAME_POLY[0])); //convert array to vector
    vector<vector<Point> > frame_contours;
    frame_contours.push_back(frame_poly_vec);
    drawContours(frame_mask, frame_contours, -1, Scalar::all(BLACK), CV_FILLED, 8, noArray(), INT_MAX, Point(0,0) );
    src &= frame_mask;
    return;
}

void maskRope(Mat& src)
{
    Mat rope_mask(NROWS, NCOLS, CV_8UC1, Scalar::all(WHITE));
    vector<Point> rope_poly_vec(ROPE_POLY, ROPE_POLY + sizeof(ROPE_POLY) / sizeof(ROPE_POLY[0])); //convert array to vector
    vector<vector<Point> > rope_contours;
    rope_contours.push_back(rope_poly_vec);
    drawContours(rope_mask, rope_contours, -1, Scalar::all(BLACK), CV_FILLED, 8, noArray(), INT_MAX, Point(0,0) );
    src &= rope_mask;
    return;
}


void extractWhitecaps(Mat& src, OpData& data)
{
    Mat whitecaps(NROWS, NCOLS, CV_8UC1, Scalar::all(BLACK));
    Mat temp(NROWS, NCOLS, CV_8UC1, Scalar::all(BLACK));
    vector<vector<Point> > contours;
    int i;

    medianBlur(src, temp, 15);

    int n_subimgs = 16; //must be a power of 2, higher values favour smaller whitecaps, lower values larger whitecaps
    optimalThreshSubimgs(temp, temp, 40, n_subimgs, n_subimgs);

    showImg("adaptive_thresh_subimgs", temp);

    int morph_size = 9;
    Mat element = getStructuringElement(2, Size(2*morph_size+1, 2*morph_size+1), Point(morph_size, morph_size));
    morphologyEx(temp, temp, MORPH_CLOSE, element);

    maskObjects(temp);

    showImg("whitecap close", temp);

    Point cent = Point(1035, 1115); //ellipse parameters found via trial and error
    int radius_x = 833;
    int radius_y = 773;
    ellipse(temp, cent, Size(radius_x, radius_y), 0, 120, 420, Scalar(WHITE), 2, 8, 0); //connects all sky regions with arc. Also means that any objects on border are removed.

    showImg("whitecap + ellipse", temp);

    findContours(temp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point());

    temp = Scalar::all(BLACK);

// ---------- FILTER CONTOURS ----------

    sort(contours.begin(), contours.end(), contourSort()); //sort contours according to area (ascending order)
    int min_contour_size = 3000; //filter out smaller contours, these are most likely rain drops
    while(contourArea(contours[0], false) < min_contour_size)
            contours.erase(contours.begin());

    contours.pop_back(); //remove largest contour, the sky

// ---------- CALCULATE AVERAGE CONTOUR PEAK GREY LEVEL ----------
    vector<int> peaks;
    int n_contours = contours.size();

    if(n_contours == 0) //if there are no contours then there is no point trying to process them.
    {
        data.addField("whitecaps", 0);
        cout << "No whitecaps found." << endl;
        return;
    }

    Mat blur;
    medianBlur(src, blur, 13);

    for(int i=0; i<n_contours; i++)
    {
        Mat hist;
        Mat contour_mask(NROWS, NCOLS, CV_8UC1, Scalar::all(BLACK));
        drawContours(contour_mask, contours, i, Scalar(WHITE), -1, 8, noArray(), INT_MAX, Point());
        getHist(blur, hist, contour_mask);
        peaks.push_back(getHistPeak(hist));
    }

    int peak_sum = 0;
    int peak_avg;
    for(int i=0; i<n_contours; i++)
        peak_sum += peaks[i];

    peak_avg = peak_sum/n_contours;

    int peak_min = 50;         ///*TODO* Get a less arbitrary minimum peak level

    vector<vector<Point> > contours_temp;
    for(int i=0; i<contours.size(); i++)
    {
        Mat hist;
        Mat contour_mask(NROWS, NCOLS, CV_8UC1, Scalar::all(BLACK));
        drawContours(contour_mask, contours, i, Scalar(WHITE), -1, 8, noArray(), INT_MAX, Point());
        getHist(blur, hist, contour_mask);
        int peak = getHistPeak(hist);

        if((peak >= peak_avg*0.80) && (peak > peak_min)) // only keep above average intensity contours
        {
            if(checkContourCorners(&contours[i]))
                //cout << "Corner found at i=" << i << endl;
                ;
            else
            {
               // cout << "No corner found at i=" << i << endl;
                contours_temp.push_back(contours[i]);
            }
        }
    }
    contours = contours_temp;

    if(contours.size() == 0) //if there are no contours then there is no point trying to process them.
    {
        data.addField("whitecaps", 0);
        cout << "No whitecaps found." << endl;
        return;
    }


// ---------- IMPROVE CONTOURS ----------

    whitecaps = Scalar(BLACK);
    drawContours(whitecaps, contours, -1, Scalar(WHITE), -1, 8, noArray(), INT_MAX, Point());

    morph_size = (src.rows/n_subimgs)/2; //large morph size joins contours that are further apart. Should join any contours that are approx 2*morph_size apart, in this case covering a missing subimg
    element = getStructuringElement(MORPH_ELLIPSE, Size(2*morph_size+1, 2*morph_size+1), Point(morph_size, morph_size));
    dilate(whitecaps, whitecaps, element); // increases whitecap roi, and joins close whitecap regions

    vector<vector<Point> > contours_clean;
    vector<vector<Point> > whitecap_contours;

    maskObjects(whitecaps);
    findContours(whitecaps, whitecap_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point());
    drawContours(whitecaps, whitecap_contours, -1, Scalar(WHITE), -1, 8, noArray(), INT_MAX, Point());

    showImg("contour dilate", whitecaps);

    for(int i=0; i<whitecap_contours.size(); i++)
    {
        Rect contour_roi = boundingRect(whitecap_contours[i]);

        int n_subimgs;
        int contour_size = contourArea(whitecap_contours[i], false);
        if(contour_size < min_contour_size)
            continue;

        if(contour_size < 15000)
            n_subimgs = 4;
        else
            n_subimgs = 8;

        int subimg_size;
        if(contour_roi.width > contour_roi.height)
            subimg_size = contour_roi.width / n_subimgs;
        else
            subimg_size = contour_roi.height / n_subimgs;

        int m, n; //round contour_roi dimensions to nearest multiple of 'n' as image size must be divisible by number of subimgs

        contour_roi.width = contour_roi.width + subimg_size - contour_roi.width%subimg_size;
        n = contour_roi.width / subimg_size;

        contour_roi.height = contour_roi.height + subimg_size - contour_roi.height%subimg_size;
        m = contour_roi.height / subimg_size;

       // cout << contour_roi.width << "x" << contour_roi.height << endl;
       // cout << n << "x" << m << endl;

        Mat object_mask = Mat(NROWS, NCOLS, CV_8UC1, Scalar(WHITE));
        maskObjects(object_mask);

        Mat contour_rect = src(Rect(contour_roi));
        Mat contour_mask = contour_rect.clone();

        medianBlur(contour_rect, contour_mask, 9);
        int min_fg_bg_diff = 20; //higher values prevent adding false 'straight line' whitecap areas, but can also cut holes in larger whitecap regions
        optimalThreshSubimgs(contour_mask, contour_mask, min_fg_bg_diff, n, m);

        contour_mask &= object_mask(Rect(contour_roi));

        vector<vector<Point> > contours_temp;
        findContours(contour_mask, contours_temp, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(contour_roi.x, contour_roi.y));

        sort(contours_temp.begin(), contours_temp.end(), contourSort());

        //Draws only the largest contour
        drawContours(contour_mask, contours_temp, -1, Scalar(WHITE), -1, 8, noArray(), INT_MAX, Point());

        morph_size = 11; //this step fixes any holes in the contours
        element = getStructuringElement(MORPH_ELLIPSE, Size(2*morph_size+1, 2*morph_size+1), Point(morph_size, morph_size));
        morphologyEx(contour_mask, contour_mask, MORPH_CLOSE, element);

        findContours(contour_mask, contours_temp, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(contour_roi.x, contour_roi.y));
        sort(contours_temp.begin(), contours_temp.end(), contourSort());

        contour_size = contourArea(contours_temp[contours_temp.size() -1], false);
        if(contour_size > min_contour_size)
            contours_clean.push_back(contours_temp[contours_temp.size() -1]);
    }

// ---------- COMPARISON OF BEFORE AND AFTER IMPROVE CONTOURS ----------

    Mat contours_before = overlayContours(src, &contours);
    Mat contours_after = overlayContours(src, &contours_clean);

    showImg("contours_before", contours_before);
    showImg("contours_after", contours_after);

    data.addImg("whitecaps", contours_after);   //add image to OpData to be saved later

    extractContourData(contours_clean, data);

    return;
}

void extractContourData(vector<vector<Point> >& contours, OpData& data)
{
    vector<Point> centroids;
    findContourCentroids(contours, centroids);

    data.addField("whitecaps", (int)contours.size());

    for(int i=0; i< contours.size(); i++)
    {
        RotatedRect bounding_rect = minAreaRect(contours[i]);
        int contour_size = contourArea(contours[i], false); //number of pixels in contour
        Point contour_centroid = centroids[i]; // centroid of contours convex hull
        int rect_width = bounding_rect.size.width; //width of bounding rect
        int rect_height = bounding_rect.size.height; //height of bounding rect
        float rect_angle = roundf(bounding_rect.angle*10)/10 + 90; //clockwise angle offset from x-axis of bounding rect, rounded to 1 decimal place
        Point2f rect_corners[4];
        bounding_rect.points(rect_corners); // corner points of bounding rect

        string centroid_string;
        ostringstream convert_centroid_x;
        ostringstream convert_centroid_y;
        convert_centroid_x << contour_centroid.x;
        convert_centroid_y << contour_centroid.y;
        centroid_string = "(" + convert_centroid_x.str() + "," + convert_centroid_y.str() + ")";

        string corners_string;
        for(int j=0; j<4; j++)
        {
            ostringstream convert_x;
            ostringstream convert_y;
            convert_x << roundf(rect_corners[j].x);
            convert_y << roundf(rect_corners[j].y);
            corners_string += "(" + convert_x.str() + "," + convert_y.str() + ")";
            if(j != 3)
                corners_string += ",";
        }


        ostringstream  convert_no;
        convert_no << i;
        string convert_no_str = convert_no.str();

        data.addField("whitecap_" + convert_no_str +  "_size", contour_size);
        data.addField("whitecap_" + convert_no_str +  "_rect_width", rect_width);
        data.addField("whitecap_" + convert_no_str +  "_rect_height", rect_height);
        data.addField("whitecap_" + convert_no_str +  "_rect_angle", rect_angle);
        data.addField("whitecap_" + convert_no_str +  "_rect_corners", corners_string);
        data.addField("whitecap_" + convert_no_str +  "_hull_centroid", centroid_string);

        cout << "Whitecap " << convert_no.str() << ", size: " << contour_size << ", rect dimensions: " << rect_width << "x" << rect_height << " at "
             << rect_angle << " degrees" << " ,corners: " << corners_string << ", hull centroid: " << centroid_string << endl;
    }

    return;
}

void findContourCentroids(vector<vector<Point> >& contours, vector<Point>& output_array)
{
    output_array.clear();

    vector<vector<Point> > hull(contours.size());
    vector<Moments> mu(contours.size());

    for(int i=0; i<contours.size(); i++)
    {
        convexHull(contours.at(i), hull[i], false, true);
        mu[i] = moments(hull[i], false);
        output_array.push_back(Point( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00));
    }
    return;
}

void showMaskOverlay(Mat& src, Mat& mask)
{
    Mat op;
    Mat temp = mask.clone();

    cvtColor(src, op, CV_GRAY2BGR);

    vector<vector<Point> > contours;
    findContours(temp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point());
    drawContours(op, contours, -1, Scalar(0,0,255), 2, 8, noArray(), INT_MAX, Point());

    showImg("showMaskOverlay", op);
    return;
}
void findSkeleton(Mat& src, Mat& dst)
{
    Mat skel(src.size(), CV_8UC1, Scalar(BLACK));
    Mat temp(src.size(), CV_8UC1);
    Mat element = getStructuringElement(MORPH_CROSS, cv::Size(3, 3));
    bool done;

    do
    {
        morphologyEx(src, temp, cv::MORPH_OPEN, element);
        bitwise_not(temp, temp);
        bitwise_and(src, temp, temp);
        bitwise_or(skel, temp, skel);
        erode(src, src, element);

        double max;
        minMaxLoc(src, 0, &max);
        done = (max == 0);
    }while (!done);

    dst = skel.clone();
    return;
}

void convertRaw(string src_path, string op_path, string format)
{
    Mat src = imreadRaw(src_path);
    OpData output(src_path,src);


    if(string(SUPPORTED_IMG_FORMATS).find(format) != string::npos)
        output.saveImg(op_path, format);
    else
    {
        cout << "Unsupported image format. Using default: " << DEFAULT_OP_IMG_EXT << endl;
        output.saveImg(op_path);
    }
    return;
}

void divideIntoSubimgs(Mat& src, vector<Mat>* op, int n, int m)
{
    op->clear();

    int w = src.cols / n;
    int h = src.rows / m;

    Mat temp;

    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
        {
            temp = src(Rect(j*w, i*h, w, h));
            op->push_back(src(Rect(j*w, i*h, w, h)));
        }
}

void combineSubimgs(vector<Mat>* src, Mat& op, int n, int m)
{
    if(src->empty())
        return;

    int w = src->at(0).cols;
    int h = src->at(0).rows;

    op = Mat(w*m, h*n, CV_8UC1, Scalar::all(BLACK));
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
        {
           // string no_string = static_cast<ostringstream*>( &(ostringstream() << (i*n + j)) )->str();
           // putText(src->at(i*n + j), no_string.c_str(), Point(0,h), FONT_HERSHEY_SIMPLEX, 1, 128, 2, 8, false);
            op(Rect(j*w, i*h, h, w)) += src->at(i*n + j);
        }

    return;
}

void optimalThreshSubimgs(Mat& src, Mat& dst, int min_fg_bg_diff, int n, int m)
{
    vector<Mat> subimgs;
    vector<int> subimgs_to_remove;
    divideIntoSubimgs(src, &subimgs, n, m);

    for(int i=0; i<subimgs.size(); i++)
    {
        Mat hist_fg;
        Mat hist_bg;
        Mat fg_mask;
        Mat bg_mask;

        bool ignore_subimg = false;

        threshold(subimgs[i], fg_mask, 0, WHITE, THRESH_BINARY | CV_THRESH_OTSU);
        bg_mask = WHITE - fg_mask;
        getHist(subimgs[i], hist_fg, fg_mask);
        getHist(subimgs[i], hist_bg, bg_mask);
        int peak_fg = getHistPeak(hist_fg);
        int peak_bg = getHistPeak(hist_bg);

        if((peak_fg - peak_bg) < min_fg_bg_diff)
            ignore_subimg = true;


// this part is a work in progress, used to ignore subimages if they have straight black/ white corners to neighbouring sub-images

    /*
        int max_col = subimgs[0].cols - 1;
        int max_row = subimgs[0].rows - 1;


        int n_corner_pixels = 3; //how many pixels either side of the corner to test
        int current_boundary_len = 0;

        bool first_row = false, last_row = false, first_col = false, last_col = false;

        if(floor((float)i/(float)n) == 0)
            first_row = true;
        if(floor((float)i/(float)n) == m-1)
            last_row = true;
        if(i%n == 0)
            first_col = true;
        if(i%n == n-1)
            last_col = true;

        if(!ignore_subimg && !first_row && !first_col) //test top left
        {
            bool no_corner = false;
            for(int j=0; j<n_corner_pixels; j++) //test left
            {
                if(!((subimgs[i].at<uchar>(0,j) == WHITE) && (subimgs[i-1].at<uchar>(max_col,j) == BLACK)))
                    no_corner = true;
            }

            for(int j=0; j<n_corner_pixels; j++) //test top
            {
                if(!((subimgs[i].at<uchar>(j,0) == WHITE) && (subimgs[i-n].at<uchar>(j,max_row) == BLACK)))
                    no_corner = true;
            }

            if(!no_corner)
            {
                cout << "White top left corner at i=" << i << endl;
                ignore_subimg = true;
            }
        }

    if(!ignore_subimg && !first_row && !last_col) //test top right
        {
            bool no_corner = false;
            for(int j=0; j<n_corner_pixels; j++) //test right
                if(!((subimgs[i].at<uchar>(last_col,j) == WHITE) && (subimgs[i+1].at<uchar>(0,j) == BLACK)))
                    no_corner = true;
            for(int j=0; j<n_corner_pixels; j++) // test top
                if(!((subimgs[i].at<uchar>(last_col-j,0) == WHITE) && (subimgs[i-n].at<uchar>(last_col-j,max_row) == BLACK)))
                    no_corner = true;

            if(!no_corner)
            {
                cout << "White top right corner at i=" << i << endl;
                ignore_subimg = true;
            }
        }

        if(!ignore_subimg && !last_row && !first_col) //test bottom left
        {
        }

        if(!ignore_subimg && !last_row && !last_col) //test bottom right
        {
        }
*/

        if(ignore_subimg)
            subimgs_to_remove.push_back(i);

        subimgs[i] = fg_mask;
    }

    for(int i=0; i<subimgs_to_remove.size(); i++)
        subimgs[subimgs_to_remove[i]] = Scalar(BLACK);

    combineSubimgs(&subimgs, dst, n, m);
    return;
}

void getHist(Mat& src, Mat& dst, Mat& mask)
{
    int bins = 256;
    float range[] = {0, 256};
    const float* ranges[] = {range};

    calcHist(&src, 1, 0, mask, dst, 1, &bins, ranges, true, false);
    return;
}

void showHist(Mat& hist)
{
    Mat hist_norm;

    int bins = hist.rows;
    int hist_w = 1024; int hist_h = 800;
    int bin_w = cvRound((double) hist_w/bins);

    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(BLACK));
    normalize(hist, hist_norm, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for(int i=1; i<bins; i++)
    {
      line(histImage, Point(bin_w*(i-1), hist_h-cvRound(hist_norm.at<float>(i-1))), Point(bin_w*(i), hist_h-cvRound(hist_norm.at<float>(i))), Scalar(WHITE), 2, 8, 0);
    }
    showImg("Histogram", histImage);
    return;
}

float getHistVariance(Mat& hist)
{

    int hist_size = hist.rows;
    float variance = 0;
    float mean = 0;
    float sum = 0;
    float mean_diff_sum = 0;

    Mat hist_norm;
    normalize(hist, hist_norm, 0, hist_size, NORM_MINMAX, -1, Mat());

    for(int i=0; i<hist_size; i++)
        sum += hist_norm.at<float>(i);

    mean = sum/hist_size;

    for(int i=0; i<hist_size; i++)
        mean_diff_sum += pow((hist_norm.at<float>(i) - mean), 2);

    variance = mean_diff_sum/hist_size;

    return variance;
}

int getHistPeak(Mat& hist)
{
    int peak_bin = 0;
    int peak_val = 0;

    for(int i=0; i<hist.rows; i++)
        if(hist.at<float>(i) > peak_val)
        {
            peak_val = hist.at<float>(i);
            peak_bin = i;
        }

        return peak_bin;
}

Mat overlayContours(Mat& src, vector<vector<Point> >* contours)
{
    Mat src_overlay;
    cvtColor(src, src_overlay, CV_GRAY2BGR);

    vector<Point> centroids;
    findContourCentroids(*contours, centroids);

    drawContours(src_overlay, *contours, -1, Scalar(0,0,255), 2, 8, noArray(), INT_MAX, Point());
    for(int i=0; i<centroids.size(); i++)
    {
        circle(src_overlay, centroids[i], 5, Scalar(255,0,0), -1, 8, 0);

        RotatedRect bounding_rect = minAreaRect(contours->at(i));
        Point2f rect_corners[4];
        bounding_rect.points(rect_corners); // corner points of bounding rect
        for(int j=0; j<4; j++)
            line(src_overlay, rect_corners[j], rect_corners[(j+1)%4], Scalar(0,255,0), 2, 8, 0);

    }

    return src_overlay;
}

bool checkContourCorners(vector<Point>* contour)
{
    bool found_corner = false;
    int n_line_points = 20; //number of consecutive points in a line to constitute a straight edge of the contour
    int x_line_points = 0;
    int y_line_points = 0;
    Point cur_point;
    Point prev_point;
    int contour_size = contour->size();

    for(int i=1; i<contour->size()*2; i++) //go around contour twice in so that it doesn't matter where on the contour we started.
    {
        if(contour->at(i%contour_size).x == contour->at((i-1)%contour_size).x)
            x_line_points++;
        else if(contour->at(i%contour_size).y != contour->at((i-1)%contour_size).y)
            x_line_points = 0;

        if(contour->at(i%contour_size).y == contour->at((i-1)%contour_size).y)
            y_line_points++;
        else if(contour->at(i%contour_size).x != contour->at((i-1)%contour_size).x)
            y_line_points = 0;

        if((y_line_points >= n_line_points) && (x_line_points >= n_line_points))
        {
            found_corner = true;
            break;
        }
    }

    return found_corner;
}
