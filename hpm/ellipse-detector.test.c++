#include <hpm/ellipse-detector.h++>
#include <hpm/util.h++>

#include <boost/ut.hpp> //import boost.ut;

using namespace hpm;

auto main() -> int {
  using namespace boost::ut;

  double const focalLength{3000.0};
  double const sphereRadius{10.0};
  PixelPosition const imageCenter{10000, 10000};

  "center sphere position"_test = [&] {
    double const zDist{1000.0};

    double const gamma1{asin(sphereRadius / zDist)};
    double const closerZ{zDist - sphereRadius * sin(gamma1)};
    double const closerR{sphereRadius * cos(gamma1)};
    double const projectionHeight{focalLength * 2 * closerR / closerZ};

    auto const gotPosition =
        hpm::Ellipse{imageCenter, projectionHeight, projectionHeight, 0.0}
            .toPosition(focalLength, imageCenter, sphereRadius * 2);

    expect(gotPosition.x == 0.0_d);
    expect(gotPosition.y == 0.0_d);
    expect(gotPosition.z == 1000.0_d);
  };

  "x-offset sphere position"_test = [&] {
    CameraFramedPosition const knownPos{10.0, 0.0, 1000.0};
    auto const [width, height, xt, yt] =
        sphereToEllipseWidthHeight(knownPos, focalLength, sphereRadius);

    auto const gotPosition =
        hpm::Ellipse{imageCenter + PixelPosition{xt, yt}, width, height, 0.0}
            .toPosition(focalLength, imageCenter, sphereRadius * 2);

    expect(gotPosition.x == 10.0_d);
    expect(gotPosition.y == 0.0_d);
    expect(gotPosition.z == 1000.0_d);
  };

  "y-offset sphere position"_test = [&] {
    CameraFramedPosition const knownPos{0.0, 10.0, 1000.0};
    auto const [width, height, xt, yt] =
        sphereToEllipseWidthHeight(knownPos, focalLength, sphereRadius);

    auto const gotPosition =
        hpm::Ellipse{imageCenter + PixelPosition{xt, yt}, width, height,
                     M_PI / 2}
            .toPosition(focalLength, imageCenter, sphereRadius * 2);

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

        auto const gotPosition =
            hpm::Ellipse{imageCenter + PixelPosition{xt, yt}, width, height,
                         ang}
                .toPosition(focalLength, imageCenter, sphereRadius * 2);

        auto constexpr EPS{0.00000000115_d}; // 1.15e-9 precision
        expect(std::abs(gotPosition.x - xDist) < EPS);
        expect(std::abs(gotPosition.y - yDist) < EPS);
        expect(std::abs(gotPosition.z - zDist) < EPS);
      }
    }
  };
  return 0;
}
