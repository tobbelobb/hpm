#pragma once

#include <opencv2/opencv.hpp>

#include <hpm/ed/EDTypes.h++>

struct EDColorConfig {
  int const gradThresh = 20;
  int const anchorThresh = 4;
  double const sigma = 1.5;
  bool const validateSegments = false;
};

struct GradientMapResult {
  cv::Mat_<short> gradImage;
  std::vector<EdgeDir> dirData;
};

class EDColor {
public:
  EDColor(cv::Mat srcImage, EDColorConfig const &config);
  cv::Mat getEdgeImage();

  std::vector<std::vector<cv::Point>> getSegments() const;
  int getSegmentNo() const;
  int getWidth() const;
  int getHeight() const;

private:
  cv::Mat edgeImage;

  int width;
  int height;

  double divForTestSegment;
  double *H;
  int np;
  int segmentNo;

  std::vector<std::vector<cv::Point>> segments;

  static size_t constexpr MIN_PATH_LEN{10};

  std::array<cv::Mat, 3> MyRGB2LabFast(cv::Mat srcImage);
  std::array<cv::Mat, 3> smoothChannels(std::array<cv::Mat, 3> src,
                                        double sigma);
  GradientMapResult
  ComputeGradientMapByDiZenzo(std::array<cv::Mat, 3> smoothLab);
  void validateEdgeSegments(std::array<cv::Mat, 3>, cv::Mat_<short> gradImage);
  void testSegment(int i, int index1, int index2, cv::Mat_<short> gradImage);
  void extractNewSegments();
  double NFA(double prob, int len);

  static void fixEdgeSegments(std::vector<std::vector<cv::Point>> map,
                              int noPixels);
};
