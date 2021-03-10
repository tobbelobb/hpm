#include <hpm/ellipse.h++>
#include <hpm/util.h++>

#include <boost/ut.hpp> //import boost.ut;

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

using namespace hpm;

auto main() -> int {
  using namespace boost::ut;

  double const focalLength{3000.0};
  double const diskRadius{10.0};
  double const sphereRadius{10.0};
  PixelPosition const imageCenter{10000, 10000};

  "center sphere position"_test = [&] {
    double const zDist{1000.0};

    double const gamma1{asin(sphereRadius / zDist)};
    double const closerZ{zDist - sphereRadius * sin(gamma1)};
    double const closerR{sphereRadius * cos(gamma1)};
    double const projectionHeight{focalLength * 2 * closerR / closerZ};

    auto const gotPosition = toPosition(
        Ellipse{imageCenter, projectionHeight, projectionHeight, 0.0},
        focalLength, imageCenter, sphereRadius * 2, MarkerType::SPHERE);

    expect(gotPosition.x == 0.0_d);
    expect(gotPosition.y == 0.0_d);
    expect(gotPosition.z == 1000.0_d);
  };

  "x-offset sphere position"_test = [&] {
    CameraFramedPosition const knownPos{10.0, 0.0, 1000.0};
    auto const [width, height, xt, yt] =
        sphereToEllipseWidthHeight(knownPos, focalLength, sphereRadius);

    auto const gotPosition = toPosition(
        Ellipse{imageCenter + PixelPosition{xt, yt}, width, height, 0.0},
        focalLength, imageCenter, sphereRadius * 2, MarkerType::SPHERE);

    expect(gotPosition.x == 10.0_d);
    expect(gotPosition.y == 0.0_d);
    expect(gotPosition.z == 1000.0_d);
  };

  "y-offset sphere position"_test = [&] {
    CameraFramedPosition const knownPos{0.0, 10.0, 1000.0};
    auto const [width, height, xt, yt] =
        sphereToEllipseWidthHeight(knownPos, focalLength, sphereRadius);

    auto const gotPosition = toPosition(
        Ellipse{imageCenter + PixelPosition{xt, yt}, width, height, M_PI / 2},
        focalLength, imageCenter, sphereRadius * 2, MarkerType::SPHERE);

    expect(gotPosition.x == 0.0_d);
    expect(gotPosition.y == 10.0_d);
    expect(gotPosition.z == 1000.0_d);
  };

  "xy-offset sphere positions"_test = [&] {
    for (double dxy{5.0}; dxy <= 1000.0; dxy = dxy + 10.0) {
      for (double ang{0.0}; ang < 2 * M_PI; ang += M_PI / 6.0) {
        double const xDist{dxy * cos(ang)};
        double const yDist{dxy * sin(ang)};
        double const zDist{1000.0};
        CameraFramedPosition const knownPos{xDist, yDist, zDist};
        auto const [width, height, xt, yt] =
            sphereToEllipseWidthHeight(knownPos, focalLength, sphereRadius);

        auto const gotPosition = toPosition(
            Ellipse{imageCenter + PixelPosition{xt, yt}, width, height, ang},
            focalLength, imageCenter, sphereRadius * 2, MarkerType::SPHERE);

        auto constexpr EPS{0.00000000115_d}; // 1.15e-9 precision
        expect(std::abs(gotPosition.x - xDist) < EPS);
        expect(std::abs(gotPosition.y - yDist) < EPS);
        expect(std::abs(gotPosition.z - zDist) < EPS);
      }
    }
  };

  "center flat disk position"_test = [&] {
    double const zDist{1000.0};
    double const projectionWidth{2 * diskRadius * focalLength / zDist};
    double const projectionHeight{projectionWidth};
    auto const gotPosition =
        toPosition(Ellipse{imageCenter, projectionWidth, projectionHeight, 0.0},
                   focalLength, imageCenter, diskRadius * 2, MarkerType::DISK);

    expect(gotPosition.x == 0.0_d);
    expect(gotPosition.y == 0.0_d);
    expect(gotPosition.z == 1000.0_d);
  };
}