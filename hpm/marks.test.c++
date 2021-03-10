#include <hpm/marks.h++>
#include <hpm/util.h++>

#include <boost/ut.hpp> //import boost.ut;

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;

  "identify sphere-marks according to provided positions"_test = [] {
    // clang-format off
    cv::Mat const cameraMatrix = (cv::Mat_<double>(3, 3) << 3000.0,    0.0, 1000.0,
                                                               0.0, 3000.0, 1000.0,
                                                               0.0,    0.0,    1.0);
    ProvidedMarkerPositions const providedPositions{-1, -1, 0,
                                                     1, -1, 0,
                                                     2,  0, 0,
                                                     1,  1, 0,
                                                    -1,  1, 0,
                                                    -2,  -0.5, 0};
    // clang-format on
    double constexpr knownMarkerDiameter{0.032};
    auto const focalLength{cameraMatrix.at<double>(0, 0)};
    PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                    cameraMatrix.at<double>(1, 2)};
    auto constexpr PIX_DIST{100.0};
    auto const CENTER{cameraMatrix.at<double>(0, 2)};
    auto const Z0{focalLength / PIX_DIST};

    auto const MINOR{sphereToEllipseWidthHeight({0, 0, Z0}, focalLength,
                                                knownMarkerDiameter / 2)
                         .width};

    // clang-format off
    Marks marks{{{{CENTER - PIX_DIST, CENTER - PIX_DIST}, 0.0, MINOR, 0.0},
                 {{CENTER - 2*PIX_DIST, CENTER + 0.5*PIX_DIST}, 0.0, MINOR, 0.0},
                 {{CENTER + PIX_DIST, CENTER - PIX_DIST}, 0.0, MINOR, 0.0},
                 {{CENTER - PIX_DIST, CENTER + PIX_DIST}, 0.0, MINOR, 0.0},
                 {{CENTER + PIX_DIST, CENTER + PIX_DIST}, 0.0, MINOR, 0.0},
                 {{CENTER + 2*PIX_DIST, CENTER}, 0.0, MINOR, 0.0}}};
    Marks const marksCpy{marks};
    // clang-format on

    double const err =
        marks.identify(providedPositions, focalLength, imageCenter,
                       knownMarkerDiameter, MarkerType::SPHERE);
    expect(marks.m_marks[0] == marksCpy.m_marks[3]);
    expect(marks.m_marks[1] == marksCpy.m_marks[4]);
    expect(marks.m_marks[2] == marksCpy.m_marks[5]);
    expect(marks.m_marks[3] == marksCpy.m_marks[2]);
    expect(marks.m_marks[4] == marksCpy.m_marks[0]);
    expect(marks.m_marks[5] == marksCpy.m_marks[1]);
    constexpr auto EPS{0.00000000001_d}; // 1e-11 precision
    expect(err < EPS);
  };

  return 0;
}
