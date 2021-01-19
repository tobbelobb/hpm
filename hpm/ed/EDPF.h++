#pragma once

#include <hpm/ed/ED.h++>
#include <hpm/ed/EDTypes.h++>

class EDPF : public ED {
public:
  EDPF(const cv::Mat &srcImage);
  EDPF(const ED &obj);
  EDPF(const EDColor &obj);

private:
  auto makeEdgeImage() -> cv::Mat;

  // differs from base class's prewit function
  // (calculates H)
  auto ComputePrewitt3x3() -> std::pair<cv::Mat_<GradPix>, std::vector<double>>;
};
