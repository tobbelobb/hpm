#ifndef  _EDColor_
#define _EDColor_

#include <opencv2/opencv.hpp>

// Look up table size for fast color space conversion
#define LUT_SIZE (1024*4096)

// Special defines
#define EDGE_VERTICAL   1
#define EDGE_HORIZONTAL 2
#define EDGE_45         3
#define EDGE_135        4

#define MAX_GRAD_VALUE 128*256
#define EPSILON 1.0
#define MIN_PATH_LEN 10


class EDColor {
public:
	EDColor(cv::Mat srcImage, int gradThresh = 20, int anchor_thresh = 4, double sigma = 1.5, bool validateSegments=false);
	cv::Mat getEdgeImage();
	std::vector<std::vector<cv::Point>> getSegments();
	int getSegmentNo();

	int getWidth();
	int getHeight();

	cv::Mat inputImage;
private:
	uchar *L_Img;
	uchar *a_Img;
	uchar *b_Img;

	uchar *smooth_L;
	uchar *smooth_a;
	uchar *smooth_b;

	uchar *dirImg;
	short *gradImg;

	cv::Mat edgeImage;
	uchar *edgeImg;

	const uchar *blueImg;
	const uchar *greenImg;
	const uchar *redImg;

	int width;
	int height;

	double divForTestSegment;
	double *H;
	int np;
	int segmentNo;

	std::vector<std::vector<cv::Point>> segments;

	static double LUT1[LUT_SIZE + 1];
	static double LUT2[LUT_SIZE + 1];
	static bool LUT_Initialized;

	void MyRGB2LabFast();
	void ComputeGradientMapByDiZenzo();
	void smoothChannel(uchar *src, uchar *smooth, double sigma);
	void validateEdgeSegments();
	void testSegment(int i, int index1, int index2);
	void extractNewSegments();
	double NFA(double prob, int len);

	static void fixEdgeSegments(std::vector<std::vector<cv::Point>> map, int noPixels);

	static void InitColorEDLib();
};

#endif // ! _EDColor_