#include <hpm/ed/EDLib.h++>
#include <hpm/test-util.h++> // getPath

#include <boost/ut.hpp> //import boost.ut;

#include <iostream>

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;
  cv::Mat billiardColor{cv::imread(getPath("billiard.jpg"), cv::IMREAD_COLOR)};
  cv::Mat billiardGrey{
      cv::imread(getPath("billiard.jpg"), cv::IMREAD_GRAYSCALE)};

  "Billiard Segments from grey image"_test = [billiardGrey] {
    ED ed{billiardGrey,
          {.op = GradientOperator::SOBEL,
           .gradThresh = 36,
           .anchorThresh = 8,
           .scanInterval = 1,
           .blurSize = 1.0,
           .sumFlag = true}};
    expect(ed.getSegmentNo() == 415_i);
  };

  "Billiard Lines from grey image"_test = [billiardGrey] {
    EDLines edLines{billiardGrey};
    expect(edLines.getLinesNo() == 518_i);
  };

  "Billiard Lines from segments"_test = [billiardGrey] {
    ED ed{billiardGrey,
          {.op = GradientOperator::SOBEL, .gradThresh = 36, .anchorThresh = 8}};
    EDLines edLines{ed};
    expect(edLines.getLinesNo() == 518_i);
  };

  "Billiard Parameter free segments from grey image"_test = [billiardGrey] {
    EDPF edpf{billiardGrey};
    expect(edpf.getSegmentNo() == 276_i);
  };

  "Billiard Circles from grey image"_test = [billiardGrey] {
    EDCircles edCirclesFromGreyImage{billiardGrey};
    expect(edCirclesFromGreyImage.getCirclesNo() == 17_i);
  };

  "Billiard Circles from parameter free segments"_test = [billiardGrey] {
    EDPF edpf{billiardGrey};
    EDCircles edCirclesFromEdpf{edpf};
    expect(edCirclesFromEdpf.getCirclesNo() == 17_i);
  };

  "Billiard Segments from color image"_test = [billiardColor] {
    EDColor edColor{billiardColor,
                    {.gradThresh = 36,
                     .anchorThresh = 4,
                     .blurSize = 1.5,
                     .filterSegments = true}};
    expect(edColor.getNumberOfSegments() == 212_u);
  };

  "Billiard Lines from segments from color image"_test = [billiardColor] {
    EDColor edColor{billiardColor,
                    {.gradThresh = 36,
                     .anchorThresh = 4,
                     .blurSize = 1.5,
                     .filterSegments = true}};
    EDLines colorLines{edColor};
    expect(colorLines.getLinesNo() == 585_i);
  };

  "Billiard Circles from segments from color image"_test = [billiardColor] {
    EDColor testEDColor{billiardColor,
                        {.gradThresh = 36,
                         .anchorThresh = 4,
                         .blurSize = 1.5,
                         .filterSegments = true}};
    EDCircles colorCircle{testEDColor};
    expect(colorCircle.getCirclesNo() == 49_i);
  };

  "Billiard color no validate segments"_test = [billiardColor] {
    EDColor testEDColor{billiardColor,
                        {.gradThresh = 36,
                         .anchorThresh = 4,
                         .blurSize = 1.5,
                         .filterSegments = false}};
    expect(testEDColor.getNumberOfSegments() == 230_u);

    EDLines colorLine = EDLines(testEDColor);
    expect(colorLine.getLinesNo() == 571_i);

    EDCircles colorCircle{testEDColor};
    expect(colorCircle.getCirclesNo() == 48_i);
  };

  return 0;
}
