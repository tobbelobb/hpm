#include <hpm/marks.h++>

#include <algorithm>
#include <limits>

using namespace hpm;

static inline auto signed2DCross(PixelPosition const &v0,
                                 PixelPosition const &v1,
                                 PixelPosition const &v2) {
  return (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
}

static inline auto isRight(PixelPosition const &v0, PixelPosition const &v1,
                           PixelPosition const &v2) -> bool {
  return signed2DCross(v0, v1, v2) <= 0.0;
}

static void fanSort(std::vector<hpm::Ellipse> &fan) {
  const auto &pivot = fan[0];
  std::sort(std::next(std::begin(fan)), std::end(fan),
            [&pivot](hpm::Ellipse const &lhs, hpm::Ellipse const &rhs) -> bool {
              return isRight(pivot.m_center, lhs.m_center, rhs.m_center);
            });
}

double hpm::identify(std::vector<Ellipse> &marks, double const markerDiameter,
                     ProvidedMarkerPositions const &markPos,
                     double const focalLength, PixelPosition const &imageCenter,
                     MarkerType markerType) {

  if (not(marks.size() >= NUMBER_OF_MARKERS)) {
    return std::numeric_limits<double>::max();
  }

  fanSort(marks);

  std::vector<double> expectedDists;
  expectedDists.reserve(NUMBER_OF_MARKERS);
  for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
    expectedDists.emplace_back(
        cv::norm(markPos.row(static_cast<int>(i)) -
                 markPos.row(static_cast<int>((i + 1) % NUMBER_OF_MARKERS))));
  }

  std::vector<CameraFramedPosition> positions{};
  for (auto const &mark : marks) {
    positions.emplace_back(
        toPosition(mark, markerDiameter, focalLength, imageCenter, markerType));
  }
  std::vector<double> foundDists;
  foundDists.reserve(NUMBER_OF_MARKERS);
  for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
    foundDists.emplace_back(
        cv::norm(positions[i] - positions[(i + 1) % NUMBER_OF_MARKERS]));
  }

  std::vector<double> errs;
  errs.reserve(NUMBER_OF_MARKERS);
  for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
    double err{0.0};
    for (size_t j{0}; j < NUMBER_OF_MARKERS; ++j) {
      double const diff{foundDists[(i + j) % NUMBER_OF_MARKERS] -
                        expectedDists[j]};
      err += diff * diff;
    }
    errs.emplace_back(err);
  }

  auto const bestErrIdx{std::distance(
      std::begin(errs), std::min_element(std::begin(errs), std::end(errs)))};
  std::rotate(std::begin(marks), std::begin(marks) + bestErrIdx,
              std::end(marks));

  return errs[static_cast<size_t>(bestErrIdx)];
}

static auto sphereZFromSemiMinor(Ellipse const &sphereProjection,
                                 double const sphereDiameter,
                                 double focalLength) -> double {
  double const semiMinor = sphereProjection.m_minor / 2.0;
  double const sphereR = sphereDiameter / 2.0;
  double const rSmall = sphereR * focalLength /
                        sqrt(semiMinor * semiMinor + focalLength * focalLength);
  double const thetaZ = atan(semiMinor / focalLength);
  return rSmall * focalLength / semiMinor + sphereR * sin(thetaZ);
}

static auto sphereCenterRayFromZ(double const sphereDiameter,
                                 double const ellipseCenterFromImageCenter,
                                 double const z) -> double {
  double const sphereR = sphereDiameter / 2;
  return ellipseCenterFromImageCenter * (z * z - sphereR * sphereR) / (z * z);
}

auto hpm::sphereCenterRay(Ellipse const &sphereProjection,
                          double const sphereDiameter, double const focalLength,
                          PixelPosition const &imageCenter)
    -> hpm::PixelPosition {
  double const z =
      sphereZFromSemiMinor(sphereProjection, sphereDiameter, focalLength);
  PixelPosition const imageCenterToEllipseCenter =
      sphereProjection.m_center - imageCenter;
  double const c = cv::norm(imageCenterToEllipseCenter);
  double const centerRay = sphereCenterRayFromZ(sphereDiameter, c, z);
  return imageCenter + centerRay * imageCenterToEllipseCenter / c;
}

static auto sphereAngularRange(double focalLength, double semiMinor,
                               double ellipseCenterFromImageCenter,
                               double centerRay) -> std::pair<double, double> {
  double const c = ellipseCenterFromImageCenter;
  double const semiMajor =
      semiMinor * sqrt(centerRay * c / (focalLength * focalLength) + 1);
  double const closest = c - semiMajor;
  double const farthest = c + semiMajor;
  double const smallestAng = atan(closest / focalLength);
  double const largestAng = atan(farthest / focalLength);
  return {smallestAng, largestAng};
}

auto hpm::sphereProjToPosition(Ellipse const &sphereProjection,
                               double sphereDiameter, double focalLength,
                               PixelPosition const &imageCenter)
    -> hpm::CameraFramedPosition {
  // The ED ellipse detector is good at determining center and minor axes
  // of an ellipse, but very bad at determining the major axis and the rotation.
  // That made this function a bit hard to write.
  double const f = focalLength;
  double const semiMinor = sphereProjection.m_minor / 2;

  // Luckily, the z position of the sphere is determined by the
  // minor axis alone, no need for the major axis or rotation.
  double const z = sphereZFromSemiMinor(sphereProjection, sphereDiameter, f);

  // The center of the ellipse is not a projection of the center of the sphere.
  // Rather, the center of the sphere projects into a point slightly closer
  // to the center of the image, like this
  PixelPosition const imageCenterToEllipseCenter =
      sphereProjection.m_center - imageCenter;
  double const c = cv::norm(imageCenterToEllipseCenter);
  double const centerRay = sphereCenterRayFromZ(sphereDiameter, c, z);

  // The center ray and the ellipse center give us the scaling
  // factor between minor and major axis, which lets
  // us compute the angular width and angular position
  // of the cone that gets projected through the pinhole
  auto const [smallestAng, largestAng] =
      sphereAngularRange(f, semiMinor, c, centerRay);

  // The angle between the center ray and the image axis
  double const alpha = std::midpoint(largestAng, smallestAng);
  // facing disc's angular radius seen from the pinhole,
  // or "half the cone's inner angle" if you will
  double const theta = std::midpoint(largestAng, -smallestAng);

  // We know that
  //   theta = asin(r/d),
  // where r is sphereR,
  // and d is the sphere's total distance from the pinhole
  double const sphereR = sphereDiameter / 2;
  double const d = sphereR / sin(theta);

  // Extracting the xy-distance using the angle between the center ray
  // and the image axis
  double const dxy = sin(alpha) * d;

  // Since ed isn't good at finding m_projection.m_rot for spheres, let's
  // calculate the rotation based on the center point, which is more accurately
  // detected by ed.
  double const rot =
      atan2(imageCenterToEllipseCenter.y, imageCenterToEllipseCenter.x);

  return {dxy * cos(rot), dxy * sin(rot), z};
}

auto hpm::diskCenterRay(Ellipse const &diskProjection,
                        double const diskDiameter, double const focalLength,
                        PixelPosition const &imageCenter) -> PixelPosition {
  (void)diskProjection;
  (void)focalLength;
  (void)imageCenter;
  (void)diskDiameter;
  return {0.0, 0.0};
}

auto hpm::diskProjToPosition(Ellipse const &diskProjection,
                             double const diskDiameter, double focalLength,
                             PixelPosition const &imageCenter)
    -> hpm::CameraFramedPosition {
  double const f = focalLength;
  PixelPosition const imageCenterToEllipseCenter =
      diskProjection.m_center - imageCenter;
  double const factorMajor = (diskDiameter / diskProjection.m_major);
  // double const factorMinor = (diskDiameter / diskProjection.m_minor);
  return {imageCenterToEllipseCenter.x * factorMajor,
          imageCenterToEllipseCenter.y * factorMajor, f * factorMajor};
}

auto hpm::toPosition(Ellipse const &markerProjection, double markerDiameter,
                     double focalLength, hpm::PixelPosition const &imageCenter,
                     MarkerType const markerType) -> hpm::CameraFramedPosition {
  switch (markerType) {
  case MarkerType::SPHERE:
    return sphereProjToPosition(markerProjection, markerDiameter, focalLength,
                                imageCenter);
  case MarkerType::DISK:
    return diskProjToPosition(markerProjection, markerDiameter, focalLength,
                              imageCenter);
  }
  return {0.0, 0.0, 0.0};
}

auto hpm::centerRay(Ellipse const &markerProjection,
                    double const markerDiameter, double const focalLength,
                    PixelPosition const &imageCenter,
                    MarkerType const markerType) -> PixelPosition {
  switch (markerType) {
  case MarkerType::SPHERE:
    return sphereCenterRay(markerProjection, markerDiameter, focalLength,
                           imageCenter);
  case MarkerType::DISK:
    return diskCenterRay(markerProjection, markerDiameter, focalLength,
                         imageCenter);
  }
  return {0.0, 0.0};
}
