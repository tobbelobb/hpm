#include <hpm/detection-result.h++>

using namespace hpm;

auto hpm::zFromSemiMinor(double markerR, double f, double semiMinor) -> double {
  double const rSmall = markerR * f / sqrt(semiMinor * semiMinor + f * f);
  double const thetaZ = atan(semiMinor / f);
  return rSmall * f / semiMinor + markerR * sin(thetaZ);
}

auto hpm::centerRayFromZ(double c, double markerR, double z) -> double {
  return c * (z * z - markerR * markerR) / (z * z);
}

auto hpm::KeyPoint::getCenterRay(double const markerR, double const f,
                                 PixelPosition const &imageCenter) const
    -> PixelPosition {
  double const z = zFromSemiMinor(markerR, f, m_minor / 2);
  PixelPosition const imageCenterToEllipseCenter = m_center - imageCenter;
  double const c = cv::norm(imageCenterToEllipseCenter);
  double const centerRay = centerRayFromZ(c, markerR, z);
  return imageCenter + centerRay * imageCenterToEllipseCenter / c;
}

hpm::KeyPoint::KeyPoint(mEllipse const &ellipse) : m_center(ellipse.center) {
  if (ellipse.axes.width >= ellipse.axes.height) {
    m_major = 2.0 * ellipse.axes.width;
    m_minor = 2.0 * ellipse.axes.height;
    m_rot = ellipse.theta;
  } else {
    m_major = 2.0 * ellipse.axes.height;
    m_minor = 2.0 * ellipse.axes.width;
    if (ellipse.theta > 0.0) {
      m_rot = ellipse.theta - M_PI / 2.0;
    } else {
      m_rot = ellipse.theta + M_PI / 2.0;
    }
  }
}
