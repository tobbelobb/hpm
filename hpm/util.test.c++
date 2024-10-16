
#include <hpm/util.h++>

#include <boost/ut.hpp> //import boost.ut;

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;
  double const focalLength{3000.0};
  double const sphereRadius{10.0};
  double const zDist{1000.0};
  double const gamma{asin(sphereRadius / zDist)};
  double const closerZ{zDist - sphereRadius * sin(gamma)};
  double const closerR{sphereRadius * cos(gamma)};
  double const moreExactHeight{focalLength * 2 * closerR / closerZ};

  auto constexpr EPS{0.000000001_d}; // 1e-9 precision

  "sphere to sphere width and height"_test = [&] {
    CameraFramedPosition const center{0.0, 0.0, zDist};
    auto const [width, height, xt, yt] =
        sphereToEllipseWidthHeight(center, focalLength, sphereRadius);

    expect(abs(width - height) < EPS)
        << "Width and height of circle projection should be equal";

    double const minHeight{focalLength * 2 * sphereRadius / center.z};
    expect(height > minHeight)
        << "Flat circle in origin should give smaller projection than a sphere"
           "with the same radius in that position.";
    expect(abs(height - moreExactHeight) < EPS);
  };

  "sphere to ellipse width and height"_test = [&] {
    double constexpr angStep{5};
    for (double len{1.0}; len < 2001; len += 100) {
      CameraFramedPosition const firstPos{len, 0, zDist};
      auto const [firstWidth, firstHeight, xt, yt] =
          sphereToEllipseWidthHeight(firstPos, focalLength, sphereRadius);

      for (double ang{0.0}; ang < 360; ang += angStep) {
        CameraFramedPosition const pos{len * cos(ang * CV_PI / 180),
                                       len * sin(ang * CV_PI / 180), zDist};
        auto const [width, height, xt, yt] =
            sphereToEllipseWidthHeight(pos, focalLength, sphereRadius);
        expect(width > height);
        expect(abs(height - moreExactHeight) < EPS);
        expect(abs(height - firstHeight) < EPS);
        if (not(abs(width - firstWidth) < EPS)) {
          std::cout << "len=" << len << " ang=" << ang << '\n';
        }
        expect((abs(width - firstWidth) < EPS) >> fatal);
      }
    }
  };

  "sphere to ellipse width and height2"_test = [&] {
    double constexpr angStep{5};
    for (double len{1.0}; len < 3001; len += 100) {
      for (double ang{0.0}; ang < 360; ang += angStep) {
        CameraFramedPosition const pos{len * cos(ang * CV_PI / 180),
                                       len * sin(ang * CV_PI / 180), zDist};
        auto const [width, height, xt, yt] =
            sphereToEllipseWidthHeight(pos, focalLength, sphereRadius);
        auto const [width2, height2] =
            sphereToEllipseWidthHeight2(pos, focalLength, sphereRadius);
        if (abs(width - width2) >= EPS) {
          std::cout << "Error at len: " << len << " ang: " << ang
                    << " error is: " << width - width2 << '\n';
        }
        expect((abs(width - width2) < EPS));
        expect((abs(height - height2) < EPS));
      }
    }
  };
}
