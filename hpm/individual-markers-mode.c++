#include <cmath> // atan
#include <ranges>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#pragma GCC diagnostic pop

#include <hpm/blob-detector.h++>
#include <hpm/individual-markers-mode.h++>
#include <hpm/util.h++>

using namespace hpm;

auto findMarks(cv::InputArray undistortedImage, ColorBounds const &colorBounds)
    -> DetectionResult {
  auto const detectionResult{blobDetect(undistortedImage, colorBounds)};
  showImage(imageWithKeyPoints(undistortedImage, {detectionResult.red, {}, {}}),
            "reds");
  showImage(
      imageWithKeyPoints(undistortedImage, {{}, detectionResult.green, {}}),
      "greens");
  showImage(
      imageWithKeyPoints(undistortedImage, {{}, {}, detectionResult.blue}),
      "blues");
  std::cout << detectionResult.red.size() << std::endl;
  std::cout << detectionResult.green.size() << std::endl;
  std::cout << detectionResult.blue.size() << std::endl;
  return detectionResult;
}

auto findMarks(cv::InputArray undistortedImage) -> DetectionResult {
  return findMarks(undistortedImage, {});
}

auto findIndividualMarkerPositions(DetectionResult const &blobs,
                                   double const knownMarkerDiameter,
                                   double const focalLength,
                                   PixelPosition const &imageCenter)
    -> std::vector<CameraFramedPosition> {
  std::vector<CameraFramedPosition> positions{};
  positions.reserve(blobs.size());
  for (auto const &blob : blobs.red) {
    positions.emplace_back(blobToPosition(hpm::KeyPoint{blob}, focalLength,
                                          imageCenter, knownMarkerDiameter));
  }
  for (auto const &blob : blobs.green) {
    positions.emplace_back(blobToPosition(hpm::KeyPoint{blob}, focalLength,
                                          imageCenter, knownMarkerDiameter));
  }
  for (auto const &blob : blobs.blue) {
    positions.emplace_back(blobToPosition(hpm::KeyPoint{blob}, focalLength,
                                          imageCenter, knownMarkerDiameter));
  }

  return positions;
  }
