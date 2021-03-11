#include <hpm/marks.h++>
#include <hpm/util.h++>

#include <boost/ut.hpp> //import boost.ut;

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;

  auto constexpr EPS9{0.000000001_d};    // 1.0e-9 precision
  auto constexpr EPS11{0.00000000001_d}; // 1e-11 precision

  "identify sphere-marks according to provided positions"_test = [&] {
    // clang-format off
    cv::Mat const cameraMatrix = (cv::Mat_<double>(3, 3) << 3000.0,    0.0,
    1000.0,
                                                               0.0, 3000.0,
                                                               1000.0, 0.0,
                                                               0.0,    1.0);
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
    std::vector<hpm::Ellipse> marks{{{{CENTER - PIX_DIST, CENTER - PIX_DIST},
    0.0, MINOR, 0.0},
                 {{CENTER - 2*PIX_DIST, CENTER + 0.5*PIX_DIST}, 0.0, MINOR,
                 0.0},
                 {{CENTER + PIX_DIST, CENTER - PIX_DIST}, 0.0, MINOR, 0.0},
                 {{CENTER - PIX_DIST, CENTER + PIX_DIST}, 0.0, MINOR, 0.0},
                 {{CENTER + PIX_DIST, CENTER + PIX_DIST}, 0.0, MINOR, 0.0},
                 {{CENTER + 2*PIX_DIST, CENTER}, 0.0, MINOR, 0.0}}};
    auto const marksCpy{marks};
    // clang-format on

    double const err = identify(marks, knownMarkerDiameter, providedPositions,
                                focalLength, imageCenter, MarkerType::SPHERE);
    expect(marks[0] == marksCpy[3]);
    expect(marks[1] == marksCpy[4]);
    expect(marks[2] == marksCpy[5]);
    expect(marks[3] == marksCpy[2]);
    expect(marks[4] == marksCpy[0]);
    expect(marks[5] == marksCpy[1]);
    expect(err < EPS11);
  };

  double const focalLength{3000.0};
  double const markerRadius{10.0};
  PixelPosition const imageCenter{10000, 10000};

  "center sphere position"_test = [&] {
    double const zDist{1000.0};

    double const gamma1{asin(markerRadius / zDist)};
    double const closerZ{zDist - markerRadius * sin(gamma1)};
    double const closerR{markerRadius * cos(gamma1)};
    double const projectionHeight{focalLength * 2 * closerR / closerZ};

    auto const gotPosition = toPosition(
        Ellipse{imageCenter, projectionHeight, projectionHeight, 0.0},
        markerRadius * 2, focalLength, imageCenter, MarkerType::SPHERE);

    expect(std::abs(gotPosition.x - 0.0) < EPS9);
    expect(std::abs(gotPosition.y - 0.0) < EPS9);
    expect(std::abs(gotPosition.z - 1000.0) < EPS9);
  };

  "x-offset sphere position"_test = [&] {
    CameraFramedPosition const knownPos{10.0, 0.0, 1000.0};
    auto const [width, height, xt, yt] =
        sphereToEllipseWidthHeight(knownPos, focalLength, markerRadius);

    auto const gotPosition = toPosition(
        Ellipse{imageCenter + PixelPosition{xt, yt}, width, height, 0.0},
        markerRadius * 2, focalLength, imageCenter, MarkerType::SPHERE);

    expect(std::abs(gotPosition.x - 10.0) < EPS9);
    expect(std::abs(gotPosition.y - 0.0) < EPS9);
    expect(std::abs(gotPosition.z - 1000.0) < EPS9);
  };

  "y-offset sphere position"_test = [&] {
    CameraFramedPosition const knownPos{0.0, 10.0, 1000.0};
    auto const [width, height, xt, yt] =
        sphereToEllipseWidthHeight(knownPos, focalLength, markerRadius);

    auto const gotPosition = toPosition(
        Ellipse{imageCenter + PixelPosition{xt, yt}, width, height, M_PI / 2},
        markerRadius * 2, focalLength, imageCenter, MarkerType::SPHERE);

    expect(std::abs(gotPosition.x - 0.0) < EPS9);
    expect(std::abs(gotPosition.y - 10.0) < EPS9);
    expect(std::abs(gotPosition.z - 1000.0) < EPS9);
  };

  "xy-offset sphere positions"_test = [&] {
    for (double dxy{5.0}; dxy <= 1000.0; dxy = dxy + 10.0) {
      for (double ang{0.0}; ang < 2 * M_PI; ang += M_PI / 6.0) {
        double const xDist{dxy * cos(ang)};
        double const yDist{dxy * sin(ang)};
        double const zDist{1000.0};
        CameraFramedPosition const knownPos{xDist, yDist, zDist};
        auto const [width, height, xt, yt] =
            sphereToEllipseWidthHeight(knownPos, focalLength, markerRadius);

        auto const gotPosition = toPosition(
            Ellipse{imageCenter + PixelPosition{xt, yt}, width, height, ang},
            markerRadius * 2, focalLength, imageCenter, MarkerType::SPHERE);

        auto constexpr EPS{0.00000000115_d}; // 1.15e-9 precision
        expect(std::abs(gotPosition.x - xDist) < EPS);
        expect(std::abs(gotPosition.y - yDist) < EPS);
        expect(std::abs(gotPosition.z - zDist) < EPS);
      }
    }
  };

  "center flat disk position"_test = [&] {
    double const zDist{1000.0};
    double const projectionWidth{2 * markerRadius * focalLength / zDist};
    double const projectionHeight{projectionWidth};
    auto const gotPosition = toPosition(
        Ellipse{imageCenter, projectionWidth, projectionHeight, 0.0},
        markerRadius * 2, focalLength, imageCenter, MarkerType::DISK);

    expect(std::abs(gotPosition.x - 0.0) < EPS9);
    expect(std::abs(gotPosition.y - 0.0) < EPS9);
    expect(std::abs(gotPosition.z - 1000.0) < EPS9);
  };

  return 0;
}
