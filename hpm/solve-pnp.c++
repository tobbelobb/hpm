#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/calib3d.hpp>
#pragma GCC diagnostic pop

#include <hpm/solve-pnp.h++>

auto solvePnp(cv::InputArray cameraMatrix,
              cv::InputArray markerPositionsRelativeToNozzle,
              IdentifiedHpMarks const &marks) -> std::optional<SixDof> {
  if (not(marks.allIdentified())) {
    std::cerr << "solvePnp got mark set with missing marker" << std::endl;
    return {};
  }

  cv::Mat const markerPositionsMatrix =
      markerPositionsRelativeToNozzle.getMat();

  std::vector<PixelPosition> const markVec{
      marks.red0.value(),   marks.red1.value(),  marks.green0.value(),
      marks.green1.value(), marks.blue0.value(), marks.blue1.value()};

  std::vector<cv::Mat> rvecs{};
  std::vector<cv::Mat> tvecs{};
  std::vector<double> reprojectionErrors{};
  cv::solvePnPGeneric(markerPositionsMatrix, markVec, cameraMatrix,
                      cv::noArray(), rvecs, tvecs, false, cv::SOLVEPNP_IPPE,
                      cv::noArray(), cv::noArray(), reprojectionErrors);

  if (rvecs.empty() or tvecs.empty() or reprojectionErrors.empty()) {
    return {};
  }
  return {SixDof{rvecs[0], tvecs[0], reprojectionErrors[0]}};
}
