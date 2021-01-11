#pragma once

#include <opencv2/opencv.hpp>

#include <hpm/ed/EDTypes.h++>

class EDColor {
public:
  EDColor(cv::Mat srcImage, int gradThresh, int anchor_thresh, double sigma,
          bool validateSegments);
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

  std::vector<EdgeDir> dirData;
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

  static size_t constexpr LUT_SIZE{1024 * 4096};
  static double LUT1[LUT_SIZE + 1];
  static double LUT2[LUT_SIZE + 1];
  static bool LUT_Initialized;
  static size_t constexpr MIN_PATH_LEN{10};

  void MyRGB2LabFast();
  void ComputeGradientMapByDiZenzo();
  void smoothChannel(uchar *src, uchar *smooth, double sigma);
  void validateEdgeSegments();
  void testSegment(int i, int index1, int index2);
  void extractNewSegments();
  double NFA(double prob, int len);

  static void fixEdgeSegments(std::vector<std::vector<cv::Point>> map,
                              int noPixels);

  static void InitColorEDLib();
};
