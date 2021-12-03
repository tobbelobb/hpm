#pragma once

#include <hpm/marks.h++>
#include <hpm/simple-types.h++>
#include <hpm/solve-pnp.h++>

#include <hpm/warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
ENABLE_WARNINGS

void showImage(cv::InputArray image, std::string const &name);

void draw(cv::InputOutputArray image, hpm::Ellipse const &ellipse,
          cv::Scalar const &color);

void draw(cv::InputOutputArray image, std::vector<hpm::Ellipse> const &marks);

void draw(cv::InputOutputArray image, hpm::SolvePnpPoints const &points,
          hpm::Vector3d const &position);

void draw(cv::InputOutputArray image, hpm::SolvePnpPoints const &points);

cv::Mat imageWith(cv::InputArray image,
                  hpm::SolvePnpPoints const &effectorPoints,
                  hpm::SolvePnpPoints const &bedPoints,
                  hpm::Vector3d const &position);

cv::Mat imageWith(cv::InputArray image, hpm::SolvePnpPoints const &points,
                  hpm::Vector3d const &position);

namespace hpm {
namespace util {
struct EllipseProjection {
  double width;
  double height;
  double xt;
  double yt;
};
} // namespace util
} // namespace hpm

// Units of sphereCenter and sphereRadius must be the same.
// Returned values will have the same units as focalLength has.
hpm::util::EllipseProjection
sphereToEllipseWidthHeight(hpm::CameraFramedPosition const &sphereCenter,
                           double focalLength, double sphereRadius);

// An alternative, geometric derivation
std::pair<double, double>
sphereToEllipseWidthHeight2(hpm::CameraFramedPosition const &sphereCenter,
                            double focalLength, double sphereRadius);

const auto AQUA{cv::Scalar(255, 255, 0)};
const auto WHITE{cv::Scalar(255, 255, 255)};
const auto BLACK{cv::Scalar(0, 0, 0)};
const auto RED{cv::Scalar(0, 0, 255)};

template <typename t>
void reorder(std::vector<t> &v, ssize_t oldIndex, ssize_t newIndex) {
  if (oldIndex > newIndex) {
    std::rotate(v.rend() - oldIndex - 1, v.rend() - oldIndex,
                v.rend() - newIndex);
  } else {
    std::rotate(v.begin() + oldIndex, v.begin() + oldIndex + 1,
                v.begin() + newIndex + 1);
  }
}
