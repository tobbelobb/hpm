#pragma once

#include <span>

#include <opencv2/opencv.hpp>

#include <hpm/ed/EDTypes.h++>

using GradPix = short;
auto constexpr GRAD_PIX_CV_TYPE{CV_16SC1};

using LabPixSingle = uchar;
auto constexpr LAB_PIX_CV_TYPE{CV_8UC3};
using LabPix = cv::Point3_<LabPixSingle>;

using Segment = std::vector<cv::Point>;

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

  std::vector<std::vector<cv::Point>> getSegments() const;
  int getNumberOfSegments() const;
  int getWidth() const;
  int getHeight() const;

private:
  cv::Mat edgeImage;

  int width;
  int height;

  static size_t constexpr MIN_SEGMENT_LEN{10};
  std::vector<Segment> segments;

  cv::Mat MyRGB2LabFast(cv::Mat srcImage);
  void blur(cv::Mat src, double blurSize);
  GradientMapResult ComputeGradientMapByDiZenzo(cv::Mat lab);

  cv::Mat makeEdgeImage(cv::Mat lab);

  template <typename Iterator>
  void drawFilteredSegment(Iterator firstPoint, Iterator lastPoint,
                           cv::Mat edgeImageIn, cv::Mat_<GradPix> gradImage,
                           std::vector<double> const &probabilityFunctionH,
                           int numberOfSegmentPieces);
  std::vector<std::vector<cv::Point>>
  validSegments(cv::Mat edgeImage, std::vector<Segment> segmentsIn) const;

  static void fixEdgeSegments(std::vector<Segment> map, int noPixels);
};
