#include <hpm/marks.h++>
#include <hpm/util.h++>

#include <boost/ut.hpp> //import boost.ut;

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;

  "filter marks by distance"_test = [] {
    // clang-format off
    cv::Mat const cameraMatrix = (cv::Mat_<double>(3, 3) << 3000.0,    0.0, 1000.0,
                                                               0.0, 3000.0, 1000.0,
                                                               0.0,    0.0,    1.0);
    ProvidedMarkerPositions const providedPositions{-1, -1, 0,
                                                     1, -1, 0,
                                                     1,  0, 0,
                                                     1,  1, 0,
                                                    -1,  1, 0,
                                                    -1,  0, 0};
    // clang-format on
    double constexpr knownMarkerDiameter{0.032};
    auto const focalLength{cameraMatrix.at<double>(0, 0)};
    PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                    cameraMatrix.at<double>(1, 2)};
    auto constexpr PIX_DIST{100.0};
    auto const CENTER{cameraMatrix.at<double>(0, 2)};
    auto const F{cameraMatrix.at<double>(0, 0)};
    auto const Z0{F / PIX_DIST};

    auto const MARKER_SIZE_STRAIGHT{
        sphereToEllipseWidthHeight({1, 0, Z0}, focalLength,
                                   knownMarkerDiameter / 2)
            .width};
    auto const MARKER_SIZE_DIAG{
        sphereToEllipseWidthHeight({1, 1, Z0}, focalLength,
                                   knownMarkerDiameter / 2)
            .width};

    // Random false positive red detection result
    hpm::Mark const falsePositiveRed{PixelPosition{200, 500}, 40.0};
    // A bit harder to sort out false positive blue detection
    hpm::Mark const falsePositiveBlue{{(CENTER + PIX_DIST) * 1.08, CENTER},
                                      MARKER_SIZE_STRAIGHT * 1.08};

    Marks marks{{{{CENTER - PIX_DIST, CENTER - PIX_DIST}, MARKER_SIZE_DIAG},
                 {{CENTER + PIX_DIST, CENTER - PIX_DIST}, MARKER_SIZE_DIAG},
                 falsePositiveRed},
                {{{CENTER + PIX_DIST, CENTER}, MARKER_SIZE_STRAIGHT},
                 {{CENTER + PIX_DIST, CENTER + PIX_DIST}, MARKER_SIZE_DIAG}},
                {{{CENTER - PIX_DIST, CENTER - PIX_DIST}, MARKER_SIZE_DIAG},
                 falsePositiveBlue,
                 {{CENTER + PIX_DIST, CENTER}, MARKER_SIZE_STRAIGHT}}};

    marks.filterByDistance(providedPositions, focalLength, imageCenter,
                           knownMarkerDiameter);
    expect(marks.red.size() == 2_ul);
    expect(marks.red[0] != falsePositiveRed);
    expect(marks.red[1] != falsePositiveRed);
    expect(marks.blue.size() == 2_ul);
    expect(marks.blue[0] != falsePositiveBlue);
    expect(marks.blue[1] != falsePositiveBlue);
  };

  return 0;
}