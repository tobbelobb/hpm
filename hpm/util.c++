#include <hpm/util.h++>

#include <hpm/warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
ENABLE_WARNINGS

#include <cmath>
#include <string>

using namespace hpm;

void draw(cv::InputOutputArray image, Ellipse const &ellipse,
          cv::Scalar const &color) {
  int constexpr LINE_WIDTH{2};
  cv::ellipse(image, ellipse.m_center,
              cv::Size2d{ellipse.m_major / 2.0,                // NOLINT
                         ellipse.m_minor / 2.0},               // NOLINT
              ellipse.m_rot * 180.0 / M_PI, 0.0, 360.0, color, // NOLINT
              LINE_WIDTH);
  cv::circle(image, ellipse.m_center, LINE_WIDTH, color, LINE_WIDTH);
}

void draw(cv::InputOutputArray image, std::vector<hpm::Ellipse> const &marks) {
  for (auto const &mark : marks) {
    draw(image, mark, AQUA);
  }
}

void draw(cv::InputOutputArray image, SolvePnpPoints const &points,
          hpm::Vector3d const &position) {
  cv::Mat imageMat{image.getMat()};
  int constexpr LINE_WIDTH{2};
  int constexpr LINE_WIDTH_BOLD{7};
  double constexpr TEXT_OFFSET{5.0};
  double constexpr TEXT_SIZE{2.0};
  for (size_t i{0}; i < points.m_pixelPositions.size(); ++i) {
    if (not points.m_identified[i]) {
      auto const pos{points.get(i)};
      cv::circle(image, pos, LINE_WIDTH, BLACK, LINE_WIDTH_BOLD);
      cv::circle(image, pos, LINE_WIDTH, RED, LINE_WIDTH);
      cv::putText(image, std::to_string(i + 1),
                  pos + cv::Point2d(TEXT_OFFSET, TEXT_OFFSET),
                  cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, BLACK, LINE_WIDTH_BOLD);
      cv::putText(image, std::to_string(i + 1),
                  pos + cv::Point2d(TEXT_OFFSET, TEXT_OFFSET),
                  cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, RED, LINE_WIDTH);
    }
  }
  for (size_t i{0}; i < points.m_pixelPositions.size(); ++i) {
    if (points.m_identified[i]) {
      auto const pos{points.get(i)};
      cv::circle(image, pos, LINE_WIDTH, BLACK, LINE_WIDTH_BOLD);
      cv::circle(image, pos, LINE_WIDTH, RED, LINE_WIDTH);
      cv::putText(image, std::to_string(i + 1),
                  pos + cv::Point2d(TEXT_OFFSET, TEXT_OFFSET),
                  cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, BLACK, LINE_WIDTH_BOLD);
      cv::putText(image, std::to_string(i + 1),
                  pos + cv::Point2d(TEXT_OFFSET, TEXT_OFFSET),
                  cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, WHITE, LINE_WIDTH);
    }
  }
  int const posRow{imageMat.rows - 200};
  int const posCol{200};
  std::stringstream ss;
  ss << std::setprecision(2) << std::fixed << position;
  cv::putText(image, ss.str(), {posCol, posRow}, cv::FONT_HERSHEY_TRIPLEX,
              TEXT_SIZE, BLACK, LINE_WIDTH_BOLD);
  cv::putText(image, ss.str(), {posCol, posRow}, cv::FONT_HERSHEY_TRIPLEX,
              TEXT_SIZE, WHITE, LINE_WIDTH);
}

void draw(cv::InputOutputArray image, double effectorReprojectionError,
          double reprojectionErrorLimit) {
  int constexpr LINE_WIDTH{2};
  int constexpr LINE_WIDTH_BOLD{7};
  int constexpr LINE_HEIGHT{100};
  double constexpr TEXT_SIZE{2.0};
  cv::Mat imageMat{image.getMat()};
  int const posRow{200};
  int const posCol{200};
  std::stringstream ss;
  ss << "Effector reproj err: " << std::setprecision(2) << std::fixed
     << effectorReprojectionError;
  cv::putText(image, ss.str(), {posCol, posRow}, cv::FONT_HERSHEY_TRIPLEX,
              TEXT_SIZE, BLACK, LINE_WIDTH_BOLD);
  cv::putText(image, ss.str(), {posCol, posRow}, cv::FONT_HERSHEY_TRIPLEX,
              TEXT_SIZE, WHITE, LINE_WIDTH);
  if (effectorReprojectionError > reprojectionErrorLimit) {
    ss.clear();
    ss.str("");
    ss << "Warning! High reprojection error.";
    cv::putText(image, ss.str(), {posCol, posRow + LINE_HEIGHT},
                cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, BLACK, LINE_WIDTH_BOLD);
    cv::putText(image, ss.str(), {posCol, posRow + LINE_HEIGHT},
                cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, RED, LINE_WIDTH);
  }
}

void draw(cv::InputOutputArray image, double effectorReprojectionError,
          double bedReprojectionError, double reprojectionErrorLimit) {
  int constexpr LINE_WIDTH{2};
  int constexpr LINE_WIDTH_BOLD{7};
  int constexpr LINE_HEIGHT{100};
  double constexpr TEXT_SIZE{2.0};
  cv::Mat imageMat{image.getMat()};
  int const posRow{200};
  int const posCol{200};
  std::stringstream ss;
  ss << "Effector reproj err: " << std::setprecision(2) << std::fixed
     << effectorReprojectionError;
  cv::putText(image, ss.str(), {posCol, posRow}, cv::FONT_HERSHEY_TRIPLEX,
              TEXT_SIZE, BLACK, LINE_WIDTH_BOLD);
  cv::putText(image, ss.str(), {posCol, posRow}, cv::FONT_HERSHEY_TRIPLEX,
              TEXT_SIZE, WHITE, LINE_WIDTH);
  ss.clear();
  ss.str("");
  ss << "Bed reproj err: " << std::setprecision(2) << std::fixed
     << bedReprojectionError;
  cv::putText(image, ss.str(), {posCol, posRow + LINE_HEIGHT},
              cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, BLACK, LINE_WIDTH_BOLD);
  cv::putText(image, ss.str(), {posCol, posRow + LINE_HEIGHT},
              cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, WHITE, LINE_WIDTH);
  if (bedReprojectionError > reprojectionErrorLimit or
      effectorReprojectionError > reprojectionErrorLimit) {
    ss.clear();
    ss.str("");
    ss << "Warning! High reprojection error.";
    cv::putText(image, ss.str(), {posCol, posRow + 2 * LINE_HEIGHT},
                cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, BLACK, LINE_WIDTH_BOLD);
    cv::putText(image, ss.str(), {posCol, posRow + 2 * LINE_HEIGHT},
                cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, RED, LINE_WIDTH);
  }
}

void draw(cv::InputOutputArray image, hpm::SolvePnpPoints const &points) {
  cv::Mat imageMat{image.getMat()};
  int constexpr LINE_WIDTH{2};
  int constexpr LINE_WIDTH_BOLD{7};
  double constexpr TEXT_OFFSET{5.0};
  double constexpr TEXT_SIZE{2.0};
  for (size_t i{0}; i < points.m_pixelPositions.size(); ++i) {
    if (not points.m_identified[i]) {
      auto const pos{points.get(i)};
      cv::circle(image, pos, LINE_WIDTH, WHITE, LINE_WIDTH_BOLD);
      cv::circle(image, pos, LINE_WIDTH, RED, LINE_WIDTH);
      cv::putText(image, std::to_string(i + 1),
                  pos + cv::Point2d(TEXT_OFFSET, TEXT_OFFSET),
                  cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, WHITE, LINE_WIDTH_BOLD);
      cv::putText(image, std::to_string(i + 1),
                  pos + cv::Point2d(TEXT_OFFSET, TEXT_OFFSET),
                  cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, RED, LINE_WIDTH);
    }
  }
  for (size_t i{0}; i < points.m_pixelPositions.size(); ++i) {
    if (points.m_identified[i]) {
      auto const pos{points.get(i)};
      cv::circle(image, pos, LINE_WIDTH, WHITE, LINE_WIDTH_BOLD);
      cv::circle(image, pos, LINE_WIDTH, RED, LINE_WIDTH);
      cv::putText(image, std::to_string(i + 1),
                  pos + cv::Point2d(TEXT_OFFSET, TEXT_OFFSET),
                  cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, WHITE, LINE_WIDTH_BOLD);
      cv::putText(image, std::to_string(i + 1),
                  pos + cv::Point2d(TEXT_OFFSET, TEXT_OFFSET),
                  cv::FONT_HERSHEY_TRIPLEX, TEXT_SIZE, BLACK, LINE_WIDTH);
    }
  }
}

auto imageWith(cv::InputArray image, std::vector<Ellipse> const &marks)
    -> cv::Mat {
  cv::Mat imageCopy{};
  image.copyTo(imageCopy);
  draw(imageCopy, marks);
  return imageCopy;
}

auto imageWith(cv::InputArray image, SolvePnpPoints const &points,
               hpm::Vector3d const &position) -> cv::Mat {
  cv::Mat imageCopy{};
  image.copyTo(imageCopy);
  draw(imageCopy, points, position);
  return imageCopy;
}

auto imageWith(cv::InputArray image, SolvePnpPoints const &points,
               hpm::Vector3d const &position, double effectorReprojectionError,
               double reprojectionErrorLimit) -> cv::Mat {
  cv::Mat imageCopy{};
  image.copyTo(imageCopy);
  draw(imageCopy, effectorReprojectionError, reprojectionErrorLimit);
  draw(imageCopy, points, position);
  return imageCopy;
}

auto imageWith(cv::InputArray image, hpm::SolvePnpPoints const &effectorPoints,
               hpm::SolvePnpPoints const &bedPoints,
               hpm::Vector3d const &position) -> cv::Mat {
  cv::Mat imageCopy{};
  image.copyTo(imageCopy);
  draw(imageCopy, bedPoints);
  draw(imageCopy, effectorPoints, position);
  return imageCopy;
}

auto imageWith(cv::InputArray image, hpm::SolvePnpPoints const &effectorPoints,
               hpm::SolvePnpPoints const &bedPoints,
               hpm::Vector3d const &position, double effectorReprojectionError,
               double bedReprojectionError, double reprojectionErrorLimit)
    -> cv::Mat {
  cv::Mat imageCopy{};
  image.copyTo(imageCopy);
  draw(imageCopy, effectorReprojectionError, bedReprojectionError,
       reprojectionErrorLimit);
  draw(imageCopy, bedPoints);
  draw(imageCopy, effectorPoints, position);
  return imageCopy;
}

void showImage(cv::InputArray image, std::string const &name) {
  static bool userWantsMoreImages{true};
  if (userWantsMoreImages) {
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    constexpr auto SHOW_PIXELS_X{1500};
    constexpr auto SHOW_PIXELS_Y{1500};
    cv::resizeWindow(name, SHOW_PIXELS_X, SHOW_PIXELS_Y);
    cv::imshow(name, image);
    int key{0};
    do {
      key = cv::waitKey(0);
      if (key == 's') {
        cv::imwrite(name, image);
      }
      if (key == 'q') {
        userWantsMoreImages = false;
      }
    } while (key != 's' and key != 'q' and key != '\r' and key != '\n' and
             key != 141); // 141 (reverse line feed) is my numpad Enter...
  }
}

static inline auto sq(auto num) { return num * num; }

auto sphereToEllipseWidthHeight(CameraFramedPosition const &sphereCenter,
                                double const focalLength,
                                double const sphereRadius)
    -> hpm::util::EllipseProjection {
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
