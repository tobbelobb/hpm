#include <hpm/solve-pnp.h++>

#include <hpm/warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/calib3d.hpp>
ENABLE_WARNINGS

#include <vector>

using namespace hpm;

SolvePnpPoints::SolvePnpPoints(
    std::vector<hpm::Ellipse> const &marks, double const markerDiameter,
    double const focalLength, PixelPosition const &imageCenter,
    MarkerType const markerType,
    CameraFramedPosition const &expectedNormalDirection) {
  if (marks.size() != NUMBER_OF_MARKERS) {
    return;
  }

  for (size_t i{0}; i < m_pixelPositions.size() and i < marks.size(); ++i) {
    m_pixelPositions[i] = // NOLINT
        centerRay(marks[i], markerDiameter, focalLength, imageCenter,
                  markerType, expectedNormalDirection);
    m_identified[i] = true; // NOLINT
  }
}

auto SolvePnpPoints::isIdentified(size_t idx) const -> bool {
  return idx < m_identified.size() and m_identified[idx]; // NOLINT
}

auto SolvePnpPoints::get(size_t idx) const -> PixelPosition {
  return m_pixelPositions[idx]; // NOLINT
}

auto SolvePnpPoints::allIdentified() const -> bool {
  return std::all_of(m_identified.begin(), m_identified.end(), std::identity());
}

auto operator<<(std::ostream &out, SolvePnpPoints const &solvePnpPoints)
    -> std::ostream & {
  for (size_t i{0}; i < solvePnpPoints.m_pixelPositions.size(); ++i) {
    if (solvePnpPoints.isIdentified(i)) {
      out << solvePnpPoints.get(i);
    } else {
      out << '?';
    }
    out << '\n';
  }
  return out;
}

auto tryHardSolvePnp(cv::InputArray cameraMatrix,
                     cv::InputArray providedPositionsRelativeToNozzle,
                     SolvePnpPoints &points) -> std::optional<SixDof> {
  if (not points.allIdentified()) {
    return {};
  }
  std::vector<SixDof> results{};
  results.reserve(NUMBER_OF_MARKERS);

  SolvePnpPoints pointsCopy{points};
  for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
    pointsCopy.m_identified[i] = false;
    results.emplace_back(
        solvePnp(cameraMatrix, providedPositionsRelativeToNozzle, pointsCopy)
            .value());
    pointsCopy.m_identified[i] = true;
  }

  auto const idxOfExcluded{static_cast<size_t>(
      std::distance(std::begin(results),
                    std::min_element(std::begin(results), std::end(results))))};
  points.m_identified[idxOfExcluded] = false;
  return {results[idxOfExcluded]};
}

auto solvePnp(cv::InputArray cameraMatrix,
              cv::InputArray providedPositionsRelativeToNozzle,
              SolvePnpPoints const &points) -> std::optional<SixDof> {

  cv::Mat providedPositionsMatrix = providedPositionsRelativeToNozzle.getMat();
  size_t const numMarkers{static_cast<size_t>(providedPositionsMatrix.rows *
                                              providedPositionsMatrix.cols) /
                          3UL};
  providedPositionsMatrix.resize(numMarkers, 3);

  std::vector<PixelPosition> usablePixelPositions{};
  cv::Mat relevantProvidedPositions(0, 3, CV_64F);
  for (size_t i{0}; i < numMarkers; ++i) {
    if (points.isIdentified(i)) {
      usablePixelPositions.emplace_back(points.get(i));
      relevantProvidedPositions.push_back(
          providedPositionsMatrix.row(static_cast<int>(i)));
    }
  }

  std::vector<cv::Mat> rvecs{};
  std::vector<cv::Mat> tvecs{};
  std::vector<double> reprojectionErrors{};
  cv::solvePnPGeneric(relevantProvidedPositions, usablePixelPositions,
                      cameraMatrix, cv::noArray(), rvecs, tvecs, false,
                      cv::SOLVEPNP_ITERATIVE, cv::noArray(), cv::noArray(),
                      reprojectionErrors);

  if (rvecs.empty() or tvecs.empty() or reprojectionErrors.empty()) {
    return {};
  }
  if (rvecs.size() > 1) {
    std::cerr << "Error: solve-pnp found " << rvecs.size()
              << " solutions. Expected 1 solution.\n";
  }
  return {SixDof{.rotation = rvecs[0],
                 .translation = tvecs[0],
                 .reprojectionError = reprojectionErrors[0]}};
}
