#include <cmath> // atan
#include <ranges>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/calib3d.hpp> // Rodrigues
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#pragma GCC diagnostic pop

#include <hpm/hpm.h++>

static void drawKeyPoints(cv::InputArray image,
                          std::vector<cv::KeyPoint> const &keyPoints,
                          cv::InputOutputArray result) {
  const auto BLACK{cv::Scalar(0)};
  cv::drawKeypoints(image, keyPoints, result, BLACK,
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::drawKeypoints(image, keyPoints, result, BLACK,
                    cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
}

static auto imageWithKeyPoints(cv::InputArray image,
                               DetectionResult const &markers) -> cv::Mat {
  cv::Mat result{};
  drawKeyPoints(image, markers.red, result);
  drawKeyPoints(result, markers.green, result);
  drawKeyPoints(result, markers.blue, result);
  return result;
}

static void showImage(cv::InputArray image, std::string const &name) {
  cv::namedWindow(name, cv::WINDOW_NORMAL);
  constexpr auto SHOW_PIXELS_X{1500};
  constexpr auto SHOW_PIXELS_Y{1500};
  cv::resizeWindow(name, SHOW_PIXELS_X, SHOW_PIXELS_Y);
  cv::imshow(name, image);
  if (cv::waitKey(0) == 's') {
    cv::imwrite(name, image);
  }
}

auto findIndividualMarkerPositions(cv::InputArray undistortedImage,
                                   double const knownMarkerDiameter,
                                   double const focalLength,
                                   cv::Point2f const &imageCenter,
                                   bool showIntermediateImages,
                                   bool showResultImage)
    -> std::vector<CameraFramedPosition> {
  if (undistortedImage.empty()) {
    return {};
  }

  auto const blobs{blobDetect(undistortedImage)};

  if (showIntermediateImages or showResultImage) {
    showImage(imageWithKeyPoints(undistortedImage, blobs),
              "markersDetected.png");
  }

  std::vector<CameraFramedPosition> positions{};
  positions.reserve(blobs.size());
  for (auto const &blob : blobs.red) {
    positions.emplace_back(
        blobToPosition(blob, focalLength, imageCenter, knownMarkerDiameter));
  }
  for (auto const &blob : blobs.green) {
    positions.emplace_back(
        blobToPosition(blob, focalLength, imageCenter, knownMarkerDiameter));
  }
  for (auto const &blob : blobs.blue) {
    positions.emplace_back(
        blobToPosition(blob, focalLength, imageCenter, knownMarkerDiameter));
  }

  return positions;
}

auto findMarks(cv::InputArray undistortedImage) -> DetectionResult {
  return blobDetect(undistortedImage);
}

auto effectorWorldPose(SixDof const &effectorPoseRelativeToCamera,
                       SixDof const &cameraPoseRelativeToWorld) -> SixDof {

  cv::Mat const rotationMatrix = [&]() {
    cv::Mat rotationMatrix_;
    cv::Rodrigues(cameraPoseRelativeToWorld.rotation, rotationMatrix_);
    return rotationMatrix_;
  }();

  cv::Mat rotation = effectorPoseRelativeToCamera.rotation -
                     cameraPoseRelativeToWorld.rotation;
  for (int i{0}; i < rotation.rows; ++i) {
    rotation.at<double>(i) = remainder(rotation.at<double>(i), CV_PI);
  }

  const auto translation =
      rotationMatrix * effectorPoseRelativeToCamera.translation +
      cameraPoseRelativeToWorld.translation;
  const auto reprojectionError = effectorPoseRelativeToCamera.reprojectionError;
  return {rotation, translation, reprojectionError};
}
