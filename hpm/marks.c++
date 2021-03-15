#include <hpm/warnings-disabler.h++>
DISABLE_WARNINGS
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
ENABLE_WARNINGS

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
  auto const C{diskProjToPosition(diskProjection, diskDiameter, focalLength,
                                  imageCenter)};
  double const factor{focalLength / C.z};
  return {imageCenter.x + factor * C.x, imageCenter.y + factor * C.y};
  //(void)diskDiameter;
  //(void)focalLength;
  //(void)imageCenter;
  // return diskProjection.m_center;
}

std::array<double, 6>
hpm::ellipseEqInCamCoords2(hpm::Ellipse const &ellipse,
                           PixelPosition const &imageCenter) {
  double ang{ellipse.m_rot};

  hpm::PixelPosition const mid{imageCenter - ellipse.m_center};
  double const h{mid.x};
  double const k{mid.y};
  double const asqinv{4.0 / (ellipse.m_major * ellipse.m_major)};
  double const bsqinv{4.0 / (ellipse.m_minor * ellipse.m_minor)};
  double const cossq{cos(ang) * cos(ang)};
  double const sinsq{sin(ang) * sin(ang)};
  double const twocossin{2.0 * cos(ang) * sin(ang)};

  double const A{cossq * asqinv + sinsq * bsqinv};
  double const B{twocossin * asqinv - twocossin * bsqinv};
  double const C{sinsq * asqinv + cossq * bsqinv};
  double const D{-2.0 * h * cossq * asqinv - twocossin * k * asqinv -
                 2.0 * h * sinsq * bsqinv + twocossin * k * bsqinv};
  double const E{-twocossin * h * asqinv - 2.0 * k * sinsq * asqinv +
                 2.0 * twocossin * h * bsqinv - 2.0 * k * cossq * bsqinv};
  double const F{h * h * cossq * asqinv + twocossin * h * k * asqinv +
                 k * k * sinsq * asqinv + h * h * sinsq * bsqinv -
                 twocossin * h * k * bsqinv + k * k * cossq * bsqinv - 1.0};
  return {A, B, C, D, E, F};
}

std::array<double, 6>
hpm::ellipseEqInCamCoords(hpm::Ellipse const &ellipse,
                          PixelPosition const &imageCenter) {
  cv::Matx22d const R_e(cos(ellipse.m_rot), -sin(ellipse.m_rot),
                        sin(ellipse.m_rot), cos(ellipse.m_rot));
  cv::Matx22d const temp(4.0 / (ellipse.m_major * ellipse.m_major), 0.0, 0.0,
                         4.0 / ((ellipse.m_minor * ellipse.m_minor)));
  cv::Matx22d const M{R_e * (temp * R_e.t())};
  cv::Matx21d const X_0(ellipse.m_center - imageCenter);
  cv::Matx21d const C{2.0 * M * X_0};
  double const F{(X_0.t() * (M * X_0))(0, 0) - 1.0};

  // clang-format off
  return {M(0, 0),
          M(0, 1),
          M(1, 1),
          C(0, 0),
          C(1, 0),
          F};
  // clang-format on
}

auto hpm::diskProjToPosition(Ellipse const &diskProjection,
                             double const diskDiameter, double focalLength,
                             PixelPosition const &imageCenter)
    // This whole algorithm was found in
    // "Camera pose estimation with circular markers (2012)" by Joris Stork,
    // which in turn found it in "Camera Calibration with Two Arbitrary Coplanar
    // Circles (2004)" by Chen et. al.

    -> hpm::CameraFramedPosition {
  auto const [A, B, C, D, E, F] =
      ellipseEqInCamCoords(diskProjection, imageCenter);
  Eigen::Matrix<double, 3, 3> Q;
  Q(0, 0) = A;
  Q(0, 1) = B;
  Q(1, 0) = Q(0, 1);
  Q(0, 2) = -D / (2.0 * focalLength);
  Q(2, 0) = Q(0, 2);
  Q(1, 1) = C;
  Q(1, 2) = -E / (2.0 * focalLength);
  Q(2, 1) = Q(1, 2);
  Q(2, 2) = F / (focalLength * focalLength);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> eigensolver(Q);
  if (eigensolver.info() != Eigen::Success) {
    std::cout << "Could not find eigenvalues!\n";
    return {0.0, 0.0, 0.0};
  }
  auto const eigenvalues = eigensolver.eigenvalues();
  auto const eigenvectors = eigensolver.eigenvectors();
  std::array<ssize_t, 3> indices{0, 1, 2};
  ssize_t i{0};
  while (i < 3 and not(std::abs(eigenvalues[indices[0]]) >=
                           std::abs(eigenvalues[indices[1]]) and
                       eigenvalues[indices[0]] * eigenvalues[indices[1]] > 0 and
                       eigenvalues[indices[0]] * eigenvalues[indices[2]] < 0)) {
    indices = {i, (i + 1) % 3, (i + 2) % 3};
    i = i + 1;
    // satisfy |lambda_0| >= |lambda_1|
    if (not(std::abs(eigenvalues[indices[0]]) >=
            std::abs(eigenvalues[indices[1]]))) {
      std::swap(indices[0], indices[1]);
    }
    // try to satisfy lambda_0*lambda_1 > 0
    if (not(eigenvalues[indices[0]] * eigenvalues[indices[1]] > 0.0)) {
      std::swap(indices[1], indices[2]);
    }
  }
  if (not(std::abs(eigenvalues[indices[0]]) >=
              std::abs(eigenvalues[indices[1]]) and
          eigenvalues[indices[0]] * eigenvalues[indices[1]] > 0 and
          eigenvalues[indices[0]] * eigenvalues[indices[2]] < 0)) {
    std::cout << "Could not satisfy Chen et al. eqn. 16\n";
    return {0.0, 0.0, 0.0};
  }
  Eigen::Matrix<double, 3, 1> lambda{eigenvalues[indices[0]],
                                     eigenvalues[indices[1]],
                                     eigenvalues[indices[2]]};
  Eigen::Matrix<double, 3, 3> V{};
  V(0, 0) = eigenvectors.col(indices[0])[0];
  V(1, 0) = eigenvectors.col(indices[0])[1];
  V(2, 0) = eigenvectors.col(indices[0])[2];
  V(0, 1) = eigenvectors.col(indices[1])[0];
  V(1, 1) = eigenvectors.col(indices[1])[1];
  V(2, 1) = eigenvectors.col(indices[1])[2];
  V(0, 2) = eigenvectors.col(indices[2])[0];
  V(1, 2) = eigenvectors.col(indices[2])[1];
  V(2, 2) = eigenvectors.col(indices[2])[2];

  // std::cout << "The eigenvalues of Q are:\n" << eigenvalues << '\n';
  // std::cout << "The sorted eigenvalues of Q are:\n" << lambda << '\n';
  // std::cout << "Here's a matrix whose columns are eigenvectors of Q \n"
  //           << "corresponding to these eigenvalues:\n"
  //           << eigenvectors << '\n';
  // std::cout << "The sorted eigenvectors:\n" << V << '\n';

  double const diskRadius{diskDiameter / 2.0};
  std::array<double, 2> constexpr SIGNS{1.0, -1.0};
  Eigen::Matrix<double, 3, 8> Cs{};
  Eigen::Matrix<double, 3, 8> Ns{};
  ssize_t j{0};
  for (auto const s1 : SIGNS) {
    for (auto const s2 : SIGNS) {
      for (auto const s3 : SIGNS) {
        double const z0 =
            s3 * lambda[1] * diskRadius / sqrt(-lambda[0] * lambda[2]);
        Eigen::Matrix<double, 3, 1> const temp{
            s2 * (lambda[2] / lambda[1]) *
                sqrt((lambda[0] - lambda[1]) / (lambda[0] - lambda[2])),
            0.0,
            -s1 * (lambda[0] / lambda[1]) *
                sqrt((lambda[1] - lambda[2]) / (lambda[0] - lambda[2]))};

        Cs.col(j) = z0 * V * temp;

        Eigen::Matrix<double, 3, 1> const temp2{
            s2 * sqrt((lambda[0] - lambda[1]) / (lambda[0] - lambda[2])), 0.0,
            -s1 * sqrt((lambda[1] - lambda[2]) / (lambda[0] - lambda[2]))};

        Ns.col(j) = V * temp2;
        j = j + 1;
      }
    }
  }
  std::array<ssize_t, 2> valids{};
  size_t founds{0};
  for (ssize_t k{0}; k < Cs.cols(); ++k) {
    if (Cs.col(k)[2] > 0.0 and Ns.col(k)[2] < 0.0) {
      valids[founds] = k;
      founds = founds + 1;
    }
    if (founds == 2) {
      break;
    }
  }
  // The algorithm finds two pose candidates
  // For now, we simply return one of them
  return {Cs.col(valids[0])[0], Cs.col(valids[0])[1], Cs.col(valids[0])[2]};
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
