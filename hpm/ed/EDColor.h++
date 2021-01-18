#pragma once

#include <span>

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

#include <hpm/ed/EDTypes.h++>

using LabPixSingle = uint8_t;
auto constexpr LAB_PIX_CV_TYPE{CV_8UC3};
using LabPix = cv::Point3_<LabPixSingle>;

struct EDColorConfig {
  int const gradThresh = 20;
  int const anchorThresh = 4;
  double const blurSize = 1.5;
  bool const filterSegments = false;
};

struct GradientMapResult {
  cv::Mat_<GradPix> gradImage;
  std::vector<EdgeDir> dirData;
};

class EDColor {
public:
  EDColor(cv::Mat srcImage, EDColorConfig const &config);
  cv::Mat getEdgeImage(); // for testing

  std::vector<Segment> getSegments() const { return segments; }
  size_t getNumberOfSegments() const { return segments.size(); }
  int getWidth() const { return width; }
  int getHeight() const { return height; }

private:
  cv::Mat edgeImage;

  int width;
  int height;

  std::vector<Segment> segments;

  cv::Mat MyRGB2LabFast(cv::Mat srcImage);
  void blur(cv::Mat src, double blurSize);
  GradientMapResult ComputeGradientMapByDiZenzo(cv::Mat lab);

  cv::Mat makeEdgeImage(cv::Mat lab);
};
