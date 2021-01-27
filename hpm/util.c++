#include <hpm/util.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
ENABLE_WARNINGS

#include <cmath>
#include <string>

using namespace hpm;

void drawMarks(cv::InputOutputArray image,
               std::vector<hpm::Mark> const &keyPoints,
               cv::Scalar const &color) {
  int constexpr LINE_WIDTH{2};
  for (auto const &keyPoint : keyPoints) {
    cv::ellipse(image, keyPoint.m_center,
                cv::Size{static_cast<int>(keyPoint.m_major / 2.0),
                         static_cast<int>(keyPoint.m_minor / 2.0)},
                keyPoint.m_rot * 180.0 / M_PI, 0.0, 360.0, color, LINE_WIDTH);
    cv::circle(image, keyPoint.m_center, LINE_WIDTH, color, LINE_WIDTH);
  }
}

void draw(cv::InputOutputArray image, Marks const &marks) {
  const auto AQUA{cv::Scalar(255, 255, 0)};
  const auto FUCHSIA{cv::Scalar(255, 0, 255)};
  const auto YELLOW{cv::Scalar(0, 255, 255)};
  drawMarks(image, marks.red, AQUA);
  drawMarks(image, marks.green, FUCHSIA);
  drawMarks(image, marks.blue, YELLOW);
}
void draw(cv::InputOutputArray image, IdentifiedMarks const &identifiedMarks) {
  const auto WHITE{cv::Scalar(255, 255, 255)};
  int constexpr LINE_WIDTH{2};
  double constexpr TEXT_OFFSET{5.0};
  for (size_t i{0}; i < identifiedMarks.m_pixelPositions.size(); ++i) {
    auto const pos{identifiedMarks.getPixelPosition(i)};
    cv::circle(image, pos, LINE_WIDTH, WHITE, LINE_WIDTH);
    cv::putText(image, std::to_string(i + 1),
                pos + cv::Point2d(TEXT_OFFSET, TEXT_OFFSET),
                cv::FONT_HERSHEY_PLAIN, 10.0, WHITE, LINE_WIDTH);
  }
}

auto imageWith(cv::InputArray image, Marks const &marks) -> cv::Mat {
  cv::Mat imageCopy{};
  image.copyTo(imageCopy);
  draw(imageCopy, marks);
  return imageCopy;
}

auto imageWith(cv::InputArray image, IdentifiedMarks const &identifiedMarks)
    -> cv::Mat {
  cv::Mat imageCopy{};
  image.copyTo(imageCopy);
  draw(imageCopy, identifiedMarks);
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

auto getSingleChannelCopy(cv::InputArray image, int channel) -> cv::Mat {
  cv::Mat singleColorImage{};
  cv::extractChannel(image, singleColorImage, channel);
  return singleColorImage;
}

auto getRedCopy(cv::InputArray image) -> cv::Mat {
  return getSingleChannelCopy(image, 2);
}

auto getGreenCopy(cv::InputArray image) -> cv::Mat {
  return getSingleChannelCopy(image, 1);
}

auto getBlueCopy(cv::InputArray image) -> cv::Mat {
  return getSingleChannelCopy(image, 0);
}

auto getValueChannelCopy(cv::InputArray image) -> cv::Mat {
  return getSingleChannelCopy(image, 2);
}

auto getSaturationChannelCopy(cv::InputArray image) -> cv::Mat {
  return getSingleChannelCopy(image, 1);
}

auto getHueChannelCopy(cv::InputArray image) -> cv::Mat {
  return getSingleChannelCopy(image, 0);
}

auto invertedCopy(cv::InputArray image) -> cv::Mat {
  cv::Mat inverted;
  cv::bitwise_not(image, inverted);
  return inverted;
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
    -> EllipseProjection {
  long double const x{static_cast<long double>(sphereCenter.x)};
  long double const y{static_cast<long double>(sphereCenter.y)};
  long double const z{static_cast<long double>(sphereCenter.z)};
  long double const r{static_cast<long double>(sphereRadius)};
  long double const f{static_cast<long double>(focalLength)};

  // Implicit equation for the searched after ellipse is
  // ax^2 + 2b''xy + a'y^2 + 2b'x + 2by + a'' = 0
  // A ray tracing derivation, see
  // https://math.stackexchange.com/questions/1367710/perspective-projection-of-a-sphere-on-a-plane
  // leads to the following values for
  // a, b'', a', b', b, and a''
  long double const a{(-sq(y) - sq(z) + sq(r))};
  long double const bpp{x * y};
  long double const ap{(-sq(x) - sq(z) + sq(r))};
  long double const bp{x * f * z};
  long double const b{y * f * z};
  long double const app{sq(f) * (-sq(x) - sq(y) + sq(r))};

  // We then do a translation of the implicit equation, to get the
  // translation equation, which is of the form
  // ax^2 + 2b''xy + a'y^2 + a''' = 0
  // We remove the linear terms in the implicit equation,
  // in exchange for having to compute the amount of translation (xt, yt),
  // which lets us compute a'''.
  long double const yt{(-b + bpp * bp / a) / (ap - sq(bpp) / a)};
  long double const xt{(-bp - bpp * yt) / a};
  long double const appp{(a * sq(xt) + 2.0L * bpp * xt * yt + ap * sq(yt) +
                          2.0L * bp * xt + 2.0L * b * yt) +
                         app};

  // After translation, we do a rotation which brings the equation into the form
  // r0*x^2 + r1*y^2 + a''' = 0,
  // where r0 and r1 are eigenvalues of the matrix
  // | a   b''|
  // | b'' a' |
  long double const delta{a * ap - sq(bpp)};
  long double const discrepand{sqrtl(sq((a + ap) / 2.0L) - delta)};
  long double const r0{(a + ap) / 2.0L + discrepand};
  long double const r1{(a + ap) / 2.0L - discrepand};

  // From this form, it's quite straightforward to rewrite into the canonical
  // form (x/w)^2 + (y/h)^2 = 1, where w is the half width of the ellipse, and h
  // is the half height of the ellipse
  long double const width{2.0L * sqrtl(abs(appp / r0))};
  long double const height{2.0L * sqrtl(abs(appp / r1))};

  return {static_cast<double>(width), static_cast<double>(height),
          static_cast<double>(xt), static_cast<double>(yt)};
}

// An alternative, geometric derivation
auto sphereToEllipseWidthHeight2(CameraFramedPosition const &sphereCenter,
                                 double const focalLength,
                                 double const sphereRadius)
    -> std::pair<double, double> {
  long double const x{static_cast<long double>(sphereCenter.x)};
  long double const y{static_cast<long double>(sphereCenter.y)};
  long double const z{static_cast<long double>(sphereCenter.z)};
  long double const dist{static_cast<long double>(cv::norm(sphereCenter))};
  long double const r{static_cast<long double>(sphereRadius)};
  long double const f{static_cast<long double>(focalLength)};

  // The height only depends on z,
  // not on x or y
  long double const gammaC{asinl(r / z)};
  long double const closerRC{r * cosl(gammaC)};
  long double const closerZC{z - r * sinl(gammaC)};
  long double const height{f * 2.0L * closerRC / closerZC};

  // The angle between the focal line and the
  // line between the pinhole and the sphere center
  long double const midAng{atanl(sqrtl(sq(x) + sq(y)) / z)};

  // The sphere will project like a disc through the pinhole.
  // The disc is slightly closer to the pinhole than the
  // sphere center is.
  // The disc also has a slightly smaller radius.
  long double const gamma{asinl(r / dist)};
  long double const smallestAng{midAng - gamma};
  long double const largestAng{midAng + gamma};

  long double const width{f * (tanl(largestAng) - tanl(smallestAng))};
  return {static_cast<double>(width), static_cast<double>(height)};
}
