#pragma once

#include <span>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/opencv.hpp>
ENABLE_WARNINGS

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
  EDColor(const cv::Mat &srcImage, EDColorConfig const &config);
  auto getEdgeImage() -> cv::Mat; // for testing

  [[nodiscard]] auto getSegments() const -> std::vector<Segment> {
    return segments;
  }
  [[nodiscard]] auto getNumberOfSegments() const -> size_t {
    return segments.size();
  }
  [[nodiscard]] auto getWidth() const -> int { return width; }
  [[nodiscard]] auto getHeight() const -> int { return height; }

private:
  cv::Mat edgeImage;

  int width;
  int height;

  std::vector<Segment> segments;

  auto MyRGB2LabFast(cv::Mat srcImage) -> cv::Mat;
  static void blur(cv::Mat src, double blurSize);
  auto ComputeGradientMapByDiZenzo(cv::Mat lab) -> GradientMapResult;

  auto makeEdgeImage(cv::Mat lab) -> cv::Mat;
};
