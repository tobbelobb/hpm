#include <hpm/solve-pnp.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/calib3d.hpp>
ENABLE_WARNINGS

#include <vector>

using namespace hpm;

auto solvePnp(cv::InputArray cameraMatrix,
              cv::InputArray providedPositionsRelativeToNozzle,
              IdentifiedHpMarks const &marks) -> std::optional<SixDof> {

  cv::Mat providedPositionsMatrix = providedPositionsRelativeToNozzle.getMat();
  size_t const numMarkers{static_cast<size_t>(providedPositionsMatrix.rows *
                                              providedPositionsMatrix.cols) /
                          3UL};
  providedPositionsMatrix.resize(numMarkers, 3);

  std::vector<PixelPosition> usablePixelPositions{};
  cv::Mat relevantProvidedPositions(0, 3, CV_64F);
  for (size_t i{0}; i < numMarkers; ++i) {
    if (marks.isIdentified(i)) {
      usablePixelPositions.push_back(marks.getPixelPosition(i));
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
