#include <hpm/solve-pnp.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/calib3d.hpp>
ENABLE_WARNINGS

#include <vector>

using namespace hpm;

SolvePnpPoints::SolvePnpPoints(Marks const &marks, double const markerR,
                               double const f,
                               PixelPosition const &imageCenter) {
  if (marks.size() != 6) {
    return;
  }

  for (size_t i{0}; i < m_pixelPositions.size() and i < marks.size(); ++i) {
    m_pixelPositions[i] =
        marks.m_marks[i].getCenterRay(markerR, f, imageCenter);
    m_identified[i] = true;
  }
}

bool SolvePnpPoints::isIdentified(size_t idx) const {
  return idx < m_identified.size() and m_identified[idx];
}

PixelPosition SolvePnpPoints::get(size_t idx) const {
  return m_pixelPositions[idx];
}

bool SolvePnpPoints::allIdentified() const {
  return std::all_of(m_identified.begin(), m_identified.end(), std::identity());
}

std::ostream &operator<<(std::ostream &out,
                         SolvePnpPoints const &solvePnpPoints) {
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
