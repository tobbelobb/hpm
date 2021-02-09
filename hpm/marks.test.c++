#include <hpm/marks.h++>
#include <hpm/util.h++>

#include <boost/ut.hpp> //import boost.ut;

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;

  "identify marks according to provided positions"_test = [] {
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

    auto const MINOR{sphereToEllipseWidthHeight({0, 0, Z0}, focalLength,
                                                knownMarkerDiameter / 2)
                         .width};
    auto const MAJOR_STRAIGHT{
        sphereToEllipseWidthHeight({1, 0, Z0}, focalLength,
                                   knownMarkerDiameter / 2)
            .width};
    auto const MAJOR_DIAG{sphereToEllipseWidthHeight({1, 1, Z0}, focalLength,
                                                     knownMarkerDiameter / 2)
                              .width};

    // Random false positive red detection result
    hpm::Mark const falsePositiveRed{PixelPosition{200, 500}, 40.0, 50.0, 0.0};
    // Very hard to sort out false positive blue detection
    hpm::Mark const falsePositiveBlue{{CENTER + PIX_DIST * 1.01, CENTER},
                                      MAJOR_STRAIGHT * 1.01,
                                      MINOR * 1.01,
                                      0.0};

    // clang-format off
    Marks marks{{{{CENTER - PIX_DIST, CENTER - PIX_DIST}, MAJOR_DIAG, MINOR, 5.0*M_PI/4.0},
                 {{CENTER + PIX_DIST, CENTER - PIX_DIST}, MAJOR_DIAG, MINOR, 7.0*M_PI/4.0},
                 falsePositiveRed},
                {{{CENTER + PIX_DIST, CENTER}, MAJOR_STRAIGHT, MINOR, 0.0},
                 {{CENTER + PIX_DIST, CENTER + PIX_DIST}, MAJOR_DIAG, MINOR, M_PI/4.0}},
                {{{CENTER - PIX_DIST, CENTER + PIX_DIST}, MAJOR_DIAG, MINOR, 3.0*M_PI/4.0},
                 falsePositiveBlue,
                 {{CENTER - PIX_DIST, CENTER}, MAJOR_STRAIGHT, MINOR, M_PI}}};
    // clang-format on

    double const err = marks.identify(providedPositions, focalLength,
                                      imageCenter, knownMarkerDiameter);
    expect(marks.m_red.size() == 2_ul);
    expect(marks.m_red[0] != falsePositiveRed);
    expect(marks.m_red[1] != falsePositiveRed);
    expect(marks.m_blue.size() == 2_ul);
    expect(marks.m_blue[0] != falsePositiveBlue);
    expect(marks.m_blue[1] != falsePositiveBlue);
    constexpr auto EPS{0.00000000001_d}; // 1e-11 precision
    expect(err < EPS);
  };

  return 0;
}
