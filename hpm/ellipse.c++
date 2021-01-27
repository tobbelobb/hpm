#include <hpm/ellipse.h++>

using namespace hpm;

auto zFromSemiMinor(double markerR, double f, double semiMinor) -> double {
  double const rSmall = markerR * f / sqrt(semiMinor * semiMinor + f * f);
  double const thetaZ = atan(semiMinor / f);
  return rSmall * f / semiMinor + markerR * sin(thetaZ);
}

auto centerRayFromZ(double c, double markerR, double z) -> double {
  return c * (z * z - markerR * markerR) / (z * z);
}

auto hpm::Ellipse::getCenterRay(double const markerR, double const f,
                                PixelPosition const &imageCenter) const
    -> PixelPosition {
  double const z = zFromSemiMinor(markerR, f, m_minor / 2);
  PixelPosition const imageCenterToEllipseCenter = m_center - imageCenter;
  double const c = cv::norm(imageCenterToEllipseCenter);
  double const centerRay = centerRayFromZ(c, markerR, z);
  return imageCenter + centerRay * imageCenterToEllipseCenter / c;
}

hpm::Ellipse::Ellipse(mEllipse const &ellipse) : m_center(ellipse.center) {
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

static auto angularRange(double f, double semiMinor, double c, double centerRay)
    -> std::pair<double, double> {
  double const semiMajor = semiMinor * sqrt(centerRay * c / (f * f) + 1);
  double const closest = c - semiMajor;
  double const farthest = c + semiMajor;
  double const smallestAng = atan(closest / f);
  double const largestAng = atan(farthest / f);
  return {smallestAng, largestAng};
}

auto hpm::Ellipse::toPosition(double focalLength,
                              hpm::PixelPosition const &imageCenter,
                              double markerDiameter) const
    -> hpm::CameraFramedPosition {
  // The ED ellipse detector is good at determining center and minor axes
  // of an ellipse, but very bad at determining the major axis and the rotation.
  // That made this function a bit hard to write.
  double const markerR = markerDiameter / 2;
  double const f = focalLength;
  double const semiMinor = m_minor / 2;

  // Luckily, the z position of the marker is determined by the
  // minor axis alone, no need for the major axis or rotation.
  double const z = zFromSemiMinor(markerR, f, semiMinor);

  // The center of the ellipse is not a projection of the center of the marker.
  // Rather, the center of the marker projects into a point slightly closer
  // to the center of the image, like this
  PixelPosition const imageCenterToEllipseCenter = m_center - imageCenter;
  double const c = cv::norm(imageCenterToEllipseCenter);
  double const centerRay = centerRayFromZ(c, markerR, z);

  // The center ray and the ellipse center give us the scaling
  // factor between minor and major axis, which lets
  // us compute the angular width and angular position
  // of the cone that gets projected through the pinhole
  auto const [smallestAng, largestAng] =
      angularRange(f, semiMinor, c, centerRay);

  // The angle between the center ray and the image axis
  double const alpha = std::midpoint(largestAng, smallestAng);
  // facing disc's angular radius seen from the pinhole,
  // or "half the cone's inner angle" if you will
  double const theta = std::midpoint(largestAng, -smallestAng);

  // We know that
  //   theta = asin(r/d),
  // where r is markerR,
  // and d is the marker's total distance from the pinhole
  double const d = markerR / sin(theta);

  // Extracting the xy-distance using the angle between the center ray
  // and the image axis
  double const dxy = sin(alpha) * d;

  // Since ed isn't good at finding m_rot, let's calculate the rotation
  // based on the center point, which is more accurately detected by ed.
  double const rot =
      atan2(imageCenterToEllipseCenter.y, imageCenterToEllipseCenter.x);

  return {dxy * cos(rot), dxy * sin(rot), z};
}
