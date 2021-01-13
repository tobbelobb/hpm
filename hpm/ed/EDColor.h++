#pragma once

#include <span>

#include <opencv2/opencv.hpp>

#include <hpm/ed/EDTypes.h++>

using gradPix = short;

struct EDColorConfig {
  int const gradThresh = 20;
  int const anchorThresh = 4;
  double const blurSize = 1.5;
  bool const filterSegments = false;
};

struct GradientMapResult {
  cv::Mat_<gradPix> gradImage;
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

  std::vector<std::vector<cv::Point>> segments;

  std::array<cv::Mat, 3> MyRGB2LabFast(cv::Mat srcImage);
  std::array<cv::Mat, 3> blur(std::array<cv::Mat, 3> src, double blurSize);
  GradientMapResult
  ComputeGradientMapByDiZenzo(std::array<cv::Mat, 3> smoothLab);

  void filterEdgeImage(std::array<cv::Mat, 3> const &Lab);

  template <typename Iterator>
  void drawFilteredSegment(Iterator firstPoint, Iterator lastPoint,
                           cv::Mat_<gradPix> gradImage,
                           std::vector<double> const &probabilityFunctionH,
                           int numberOfSegmentPieces);
  void extractNewSegments();

  static void fixEdgeSegments(std::vector<std::vector<cv::Point>> map,
                              int noPixels);
};
