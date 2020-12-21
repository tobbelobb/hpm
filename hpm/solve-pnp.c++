#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/calib3d.hpp>
#pragma GCC diagnostic pop

#include <hpm/solve-pnp.h++>

using namespace hpm;

auto solvePnp(cv::InputArray cameraMatrix,
              cv::InputArray providedPositionsRelativeToNozzle,
              IdentifiedHpMarks const &marks) -> std::optional<SixDof> {

  cv::Mat const providedPositionsMatrix =
      providedPositionsRelativeToNozzle.getMat();

  auto const numMarkers{providedPositionsMatrix.rows};

  std::vector<PixelPosition> usablePixelPositions{};
  cv::Mat relevantProvidedPositions(0, 3, CV_64F);
  // Pipes?
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wconversion"
  for (size_t i{0}; i < numMarkers; ++i) {
    if (marks.m_identified[i]) {
      usablePixelPositions.push_back(marks.m_pixelPositions[i]);
      relevantProvidedPositions.push_back(providedPositionsMatrix.row(i));
    }
  }
#pragma GCC diagnostic pop

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
