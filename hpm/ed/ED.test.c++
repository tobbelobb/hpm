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
    EDColor testEDColor{colorImg, 36, 4, 1.5, true};
    expect(testEDColor.getSegmentNo() == 212_i);

    EDLines colorLine = EDLines(testEDColor);
    expect(colorLine.getLinesNo() == 585_i);

    EDCircles colorCircle{testEDColor};
    expect(colorCircle.getCirclesNo() == 49_i);
  };

  return 0;
}