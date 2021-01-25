
#include <boost/ut.hpp> //import boost.ut;

#include <hpm/ellipse-detector.h++>
#include <hpm/util.h++>

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
        ellipseToPosition(hpm::KeyPoint{.m_center = imageCenter,
                                        .m_major = projectionHeight,
                                        .m_minor = projectionHeight,
                                        .m_rot = 0.0},
                          focalLength, imageCenter, sphereRadius * 2);

    expect(gotPosition.x == 0.0_d);
    expect(gotPosition.y == 0.0_d);
    expect(gotPosition.z == 1000.0_d);
  };

  "x-offset sphere position"_test = [&] {
    CameraFramedPosition const knownPos{10.0, 0.0, 1000.0};
    auto const [width, height, xt, yt] =
        sphereToEllipseWidthHeight(knownPos, focalLength, sphereRadius);

    auto const gotPosition = ellipseToPosition(
        hpm::KeyPoint{.m_center = imageCenter + PixelPosition{xt, yt},
                      .m_major = width,
                      .m_minor = height,
                      .m_rot = 0.0},
        focalLength, imageCenter, sphereRadius * 2);

    expect(gotPosition.x == 10.0_d);
    expect(gotPosition.y == 0.0_d);
    expect(gotPosition.z == 1000.0_d);
  };

  "y-offset sphere position"_test = [&] {
    CameraFramedPosition const knownPos{0.0, 10.0, 1000.0};
    auto const [width, height, xt, yt] =
        sphereToEllipseWidthHeight(knownPos, focalLength, sphereRadius);

    auto const gotPosition = ellipseToPosition(
        hpm::KeyPoint{.m_center = imageCenter + PixelPosition{xt, yt},
                      .m_major = width,
                      .m_minor = height,
                      .m_rot = M_PI / 2},
        focalLength, imageCenter, sphereRadius * 2);

    expect(gotPosition.x == 0.0_d);
    expect(gotPosition.y == 10.0_d);
    expect(gotPosition.z == 1000.0_d);
  };

  "xy-offset sphere positions"_test = [&] {
    for (double dxy{10.0}; dxy <= 1000.0; dxy = dxy + 10.0) {
      for (double ang{0.0}; ang < 2 * M_PI; ang += M_PI / 6.0) {
        double const xDist{dxy * cos(ang)};
        double const yDist{dxy * sin(ang)};
        double const zDist{1000.0};
        CameraFramedPosition const knownPos{xDist, yDist, zDist};
        auto const [width, height, xt, yt] =
            sphereToEllipseWidthHeight(knownPos, focalLength, sphereRadius);

        auto const gotPosition = ellipseToPosition(
            hpm::KeyPoint{.m_center = imageCenter + PixelPosition{xt, yt},
                          .m_major = width,
                          .m_minor = height,
                          .m_rot = ang},
            focalLength, imageCenter, sphereRadius * 2);

        // TODO: Cannot for my life of it understand why precision
        // isn't better than 1e-1 with the current implementation
        // of ellipseToPosition
        auto constexpr EPS2{0.02_d}; // 2e-2 precision
        expect(std::abs(gotPosition.x - xDist) < EPS2);
        expect(std::abs(gotPosition.y - yDist) < EPS2);
        expect(std::abs(gotPosition.z - zDist) < EPS2);
      }
    }
  };
  return 0;
}
