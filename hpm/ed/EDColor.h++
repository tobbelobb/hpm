#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop

// Special defines
#define EDGE_VERTICAL 1
#define EDGE_HORIZONTAL 2
#define EDGE_45 3
#define EDGE_135 4

#define MAX_GRAD_VALUE 128 * 256
#define EPSILON 1.0
#define MIN_PATH_LEN 10

class EDColor {
public:
  EDColor(cv::Mat srcImage, int gradThresh = 20, int anchor_thresh = 4,
          double sigma = 1.5, bool validateSegments = false);
  cv::Mat getEdgeImage();
  std::vector<std::vector<cv::Point>> getSegments();

  size_t getSegmentNo() const;
  size_t getWidth() const;
  size_t getHeight() const;

  cv::Mat inputImage;

private:
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

  size_t width;
  size_t height;

  double *H;
  int np;
  int segmentNo;

  std::vector<std::vector<cv::Point>> segments;

  static size_t constexpr LUT_SIZE{1024 * 4096};

  std::array<double, LUT_SIZE + 1> getLut(int which);
  std::array<std::vector<uchar>, 3> MyRGB2LabFast();
  void ComputeGradientMapByDiZenzo();
  void smoothChannel(std::vector<uchar> &src, uchar *smooth, double sigma);
  void validateEdgeSegments();
  void testSegment(int i, int index1, int index2);
  void extractNewSegments();
  double NFA(double prob, int len);

  static void fixEdgeSegments(std::vector<std::vector<cv::Point>> map,
                              int noPixels);
};
