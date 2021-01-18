#pragma once

#include <hpm/ed/ED.h++>
#include <hpm/ed/EDTypes.h++>

class EDPF : public ED {
public:
  EDPF(cv::Mat srcImage);
  EDPF(ED obj);
  EDPF(EDColor obj);

private:
  cv::Mat makeEdgeImage();

  // differs from base class's prewit function
  // (calculates H)
  std::pair<cv::Mat_<GradPix>, std::vector<double>> ComputePrewitt3x3();
};
