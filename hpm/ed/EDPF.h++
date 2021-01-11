#pragma once

#include <hpm/ed/ED.h++>
#include <hpm/ed/EDTypes.h++>

class EDPF : public ED {
public:
  EDPF(cv::Mat srcImage);
  EDPF(ED obj);
  EDPF(EDColor obj);

private:
  double divForTestSegment;
  double *H;
  int np;
  short *gradImg;

  void validateEdgeSegments();
  short *ComputePrewitt3x3(); // differs from base class's prewit function
                              // (calculates H)
  void TestSegment(int i, int index1, int index2);
  void ExtractNewSegments();
  double NFA(double prob, int len);
};
