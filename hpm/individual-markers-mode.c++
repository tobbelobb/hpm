#include <hpm/blob-detector.h++>
#include <hpm/ellipse-detector.h++>
#include <hpm/individual-markers-mode.h++>
#include <hpm/util.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
ENABLE_WARNINGS

#include <algorithm>
#include <cmath> // atan
#include <ranges>
#include <string>

using namespace hpm;

auto find(cv::InputArray undistortedImage,
          hpm::ProvidedMarkerPositions const &markPos, double const focalLength,
          hpm::PixelPosition const &imageCenter, double const markerDiameter,
          bool showIntermediateImages, bool verbose)
    -> std::pair<hpm::IdentifiedHpMarks, hpm::DetectionResult> {
  DetectionResult const marks{findMarks(undistortedImage, markPos, focalLength,
                                        imageCenter, markerDiameter,
                                        showIntermediateImages, verbose)};
  IdentifiedHpMarks const identifiedMarks{marks, markerDiameter / 2.0,
                                          focalLength, imageCenter};
  return {identifiedMarks, marks};
}

auto findMarks(cv::InputArray undistortedImage,
               hpm::ProvidedMarkerPositions const &markPos,
               double const focalLength, PixelPosition const &imageCenter,
               double const markerDiameter, bool showIntermediateImages,
               bool verbose) -> DetectionResult {

  auto marks{ellipseDetect(undistortedImage, showIntermediateImages)};

  if (showIntermediateImages) {
    showImage(imageWith(undistortedImage, marks),
              "found-marks-before-filtering-by-distance.png");
  }
  marks.filterByDistance(markPos, focalLength, imageCenter, markerDiameter);
  if (showIntermediateImages) {
    showImage(imageWith(undistortedImage, marks),
              "found-marks-after-filtering-by-distance.png");
  }
  if (verbose) {
    std::cout << "Found " << marks.red.size() << " red markers, "
              << marks.green.size() << " green markers, and "
              << marks.blue.size() << " blue markers\n";
  }

  return marks;
}

auto findIndividualMarkerPositions(DetectionResult const &detectionResult,
                                   double const knownMarkerDiameter,
                                   double const focalLength,
                                   PixelPosition const &imageCenter)
    -> std::vector<CameraFramedPosition> {
  std::vector<CameraFramedPosition> positions{};
  positions.reserve(detectionResult.size());
  for (auto const &detected : detectionResult.red) {
    positions.emplace_back(ellipseToPosition(detected, focalLength, imageCenter,
                                             knownMarkerDiameter));
  }
  for (auto const &detected : detectionResult.green) {
    positions.emplace_back(ellipseToPosition(detected, focalLength, imageCenter,
                                             knownMarkerDiameter));
  }
  for (auto const &detected : detectionResult.blue) {
    positions.emplace_back(ellipseToPosition(detected, focalLength, imageCenter,
                                             knownMarkerDiameter));
  }

  return positions;
}
