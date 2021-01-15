#include <iostream>

#include <boost/ut.hpp> //import boost.ut;

#include <hpm/test-util.h++> // getPath

#include <hpm/ed/EDLib.h++>

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;
  cv::Mat billiardColor{cv::imread(getPath("billiard.jpg"), cv::IMREAD_COLOR)};
  cv::Mat billiardGrey{
      cv::imread(getPath("billiard.jpg"), cv::IMREAD_GRAYSCALE)};

  "Billiard Segments from grey image"_test = [billiardGrey] {
    ED ed{billiardGrey, GradientOperator::SOBEL, 36, 8, 1, 10, 1.0, true};
    expect(ed.getSegmentNo() == 415_i);
  };

  skip / "Billiard Lines from grey image"_test = [billiardGrey] {
    EDLines edLines{billiardGrey};
    expect(edLines.getLinesNo() == 518_i);
  };

  skip / "Billiard Lines from segments"_test = [billiardGrey] {
    ED ed{billiardGrey, GradientOperator::SOBEL, 36, 8, 1, 10, 1.0, true};
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
    expect(edColor.getNumberOfSegments() == 212_i);
  };

  skip /
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
    expect(testEDColor.getNumberOfSegments() == 230_i);

    EDLines colorLine = EDLines(testEDColor);
    expect(colorLine.getLinesNo() == 571_i);

    EDCircles colorCircle{testEDColor};
    expect(colorCircle.getCirclesNo() == 48_i);
  };

  return 0;
}
