#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <opencv2/calib3d.hpp>
#pragma GCC diagnostic pop

#include <hpm/solve-pnp.h++>

std::optional<SixDof> solvePnp(cv::InputArray cameraMatrix,
                               std::vector<PixelPosition> const &marks) {
  //
  // This should come all the way from the command line
  std::vector<cv::Point3d> const markersPositionsRelativeToNozzle{
      {144.896, 0.0, 0.0},       // blue on x-axis
      {72.4478, -125.483, 0.0},  // blue back
      {-144.896, 0.0, 0.0},      // green on x-axis
      {-72.4478, -125.483, 0.0}, // green back
      {72.4478, 125.483, 0.0},   // red right
      {-72.4478, 125.483, 0.0},  // red left
      {0.0, 0.0, 0.0}};

  if (marks.size() != markersPositionsRelativeToNozzle.size()) {
    std::cerr << "Found positions and known positions have different sizes: "
              << "founds: " << marks.size()
              << ", knowns: " << markersPositionsRelativeToNozzle.size()
              << std::endl;
    return {};
  }

  std::vector<cv::Mat> rvecs{};
  std::vector<cv::Mat> tvecs{};
  std::vector<double> reprojectionErrors{};
  cv::solvePnPGeneric(markersPositionsRelativeToNozzle, marks, cameraMatrix,
                      cv::noArray(), rvecs, tvecs, false, cv::SOLVEPNP_IPPE,
                      cv::noArray(), cv::noArray(), reprojectionErrors);

  if (rvecs.empty() or tvecs.empty() or reprojectionErrors.empty()) {
    return {};
  }
  return {SixDof{rvecs[0], tvecs[0], reprojectionErrors[0]}};
}
