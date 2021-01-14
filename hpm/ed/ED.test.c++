#include <iostream>

#include <boost/ut.hpp> //import boost.ut;

#include <hpm/test-util.h++> // getPath

#include <hpm/ed/EDLib.h++>

using namespace cv;
using namespace std;

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;

  "Billiard original"_test = [] {
    Mat colorImg = imread(getPath("billiard.jpg"));
    EDColor testEDColor{colorImg,
                        {.gradThresh = 36,
                         .anchorThresh = 4,
                         .blurSize = 1.5,
                         .filterSegments = true}};
    expect(testEDColor.getNumberOfSegments() == 212_i);

    EDLines const colorLine = EDLines(testEDColor);
    expect(colorLine.getLinesNo() == 585_i);

    EDCircles colorCircle{testEDColor};
    expect(colorCircle.getCirclesNo() == 49_i);
  };

  // These values are different (much better)
  // than the original program, which shows a lot
  // of noise in the no validate case.
  "Billiard no validate segments"_test = [] {
    Mat colorImg = imread(getPath("billiard.jpg"));
    EDColor testEDColor{colorImg,
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
