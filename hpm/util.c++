#include <cmath>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#pragma GCC diagnostic pop

#include <hpm/util.h++>

using namespace hpm;

void drawKeyPoints(cv::InputOutputArray image,
                   std::vector<hpm::KeyPoint> const &keyPoints,
                   cv::Scalar const &color) {
  for (auto const &keyPoint : keyPoints) {
    cv::circle(image, keyPoint.center, static_cast<int>(keyPoint.size / 2.0),
               color, 3);
    cv::circle(image, keyPoint.center, 2, color, 3);
  }
}

void drawDetectionResult(cv::InputOutputArray image,
                         DetectionResult const &markers) {
  const auto AQUA{cv::Scalar(255, 255, 0)};
  const auto FUCHSIA{cv::Scalar(255, 0, 255)};
  const auto YELLOW{cv::Scalar(0, 255, 255)};
  drawKeyPoints(image, markers.red, AQUA);
  drawKeyPoints(image, markers.green, FUCHSIA);
  drawKeyPoints(image, markers.blue, YELLOW);
}

auto imageWithDetectionResult(cv::InputArray image,
                              DetectionResult const &detectionResult)
    -> cv::Mat {
  cv::Mat imageCopy{};
  image.copyTo(imageCopy);
  drawDetectionResult(imageCopy, detectionResult);
  return imageCopy;
}

void showImage(cv::InputArray image, std::string const &name) {
  cv::namedWindow(name, cv::WINDOW_NORMAL);
  constexpr auto SHOW_PIXELS_X{1500};
  constexpr auto SHOW_PIXELS_Y{1500};
  cv::resizeWindow(name, SHOW_PIXELS_X, SHOW_PIXELS_Y);
  cv::imshow(name, image);
  if (cv::waitKey(0) == 's') {
    cv::imwrite(name, image);
  }
}

auto ScalarBGR2HSV(cv::Scalar const &bgr) -> cv::Scalar {
  cv::Mat const bgrMat{1, 1, CV_8UC3, bgr};
  cv::Mat hsv{1, 1, CV_8UC3, bgr};
  cv::cvtColor(bgrMat, hsv, cv::COLOR_BGR2HSV);
  return cv::Scalar{static_cast<double>(hsv.data[0]),
                    static_cast<double>(hsv.data[1]),
                    static_cast<double>(hsv.data[2]), 1.0};
}

static inline auto sq(auto num) { return num * num; }

auto sphereToEllipseWidthHeight(CameraFramedPosition const &sphereCenter,
                                double const focalLength,
                                double const sphereRadius)
    -> std::pair<double, double> {
  double const x{sphereCenter.x};
  double const y{sphereCenter.y};
  double const z{sphereCenter.z};
  double const r{sphereRadius};
  double const f{focalLength};

  // Implicit equation for the searched after ellipse is
  // ax^2 + 2b''xy + a'y^2 + 2b'x + 2by + a'' = 0
  // A ray tracing derivation, see
  // https://math.stackexchange.com/questions/1367710/perspective-projection-of-a-sphere-on-a-plane
  // leads to the following values for
  // a, b'', a', b', b, and a''
  double const a{(-sq(y) - sq(z) + sq(r))};
  double const bpp{x * y};
  double const ap{(-sq(x) - sq(z) + sq(r))};
  double const bp{x * f * z};
  double const b{y * f * z};
  double const app{sq(f) * (-sq(x) - sq(y) + sq(r))};

  // We then do a translation of the implicit equation, to get the
  // translation equation, which is of the form
  // ax^2 + 2b''xy + a'y^2 + a'''
  // We remove the linear terms in the implicit equation,
  // in exchange for having to compute the amount of translation (xt, yt),
  // which lets us compute a'''.
  double const yt{(-b + bpp * bp / a) / (ap - sq(bpp) / a)};
  double const xt{(-bp - bpp * yt) / a};
  double const appp{(a * sq(xt) + 2 * bpp * xt * yt + ap * sq(yt) +
                     2 * bp * xt + 2 * b * yt) +
                    app};

  // After translation, we do a rotation which brings the equation into the form
  // r0*x^2 + r1*y^2 + a''' = 0,
  // where r0 and r1 are eigenvalues of the matrix
  // | a   b''|
  // | b'' a' |
  double const delta{a * ap - sq(bpp)};
  double const discrepand{sqrt(sq((a + ap) / 2) - delta)};
  double const r0{(a + ap) / 2.0 + discrepand};
  double const r1{(a + ap) / 2.0 - discrepand};

  // From this form, it's quite straightforward to rewrite into the canonical
  // form (x/w)^2 + (y/h)^2 = 1, where w is the half width of the ellipse, and h
  // is the half height of the ellipse
  double const width{2.0 * sqrt(abs(appp / r0))};
  double const height{2.0 * sqrt(abs(appp / r1))};

  return {width, height};
}
