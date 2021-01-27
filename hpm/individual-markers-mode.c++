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

auto findMarks(cv::InputArray undistortedImage,
               hpm::ProvidedMarkerPositions const &markPos,
               double const focalLength, PixelPosition const &imageCenter,
               double const markerDiameter, bool showIntermediateImages)
    -> DetectionResult {

  auto foundMarks{ellipseDetect(undistortedImage, showIntermediateImages)};

  if (showIntermediateImages) {
    showImage(imageWith(undistortedImage, foundMarks),
              "found-marks-before-filtering-by-distance.png");
  }
  filterMarksByDistance(foundMarks, markPos, focalLength, imageCenter,
                        markerDiameter);
  if (showIntermediateImages) {
    showImage(imageWith(undistortedImage, foundMarks),
              "found-marks-after-filtering-by-distance.png");
  }

  return foundMarks;
}

void filterMarksByDistance(DetectionResult &marks,
                           hpm::ProvidedMarkerPositions const &markPos,
                           double const focalLength,
                           PixelPosition const &imageCenter,
                           double const markerDiameter) {
  auto filterSingleColor = [&](std::vector<KeyPoint> &marksOfOneColor,
                               double expectedDistance) {
    size_t const sz{marksOfOneColor.size()};
    if (sz > 2) {

      std::vector<CameraFramedPosition> allPositions{};
      for (auto const &mark : marksOfOneColor) {
        allPositions.emplace_back(
            blobToPosition(mark, focalLength, imageCenter, markerDiameter));
      }

      std::vector<std::pair<size_t, size_t>> allPairs{};
      for (size_t i{0}; i < sz; ++i) {
        for (size_t j{i + 1}; j < sz; ++j) {
          allPairs.emplace_back(i, j);
        }
      }

      std::vector<double> allDistances{};
      for (auto const &pair : allPairs) {
        allDistances.emplace_back(
            cv::norm(allPositions[pair.first] - allPositions[pair.second]));
      }

      auto const winnerPair = allPairs[static_cast<size_t>(std::distance(
          allDistances.begin(),
          std::min_element(
              allDistances.begin(), allDistances.end(),
              [expectedDistance](double distanceLeft, double distanceRight) {
                return abs(distanceLeft - expectedDistance) <
                       abs(distanceRight - expectedDistance);
              })))];

      auto const first{marksOfOneColor[winnerPair.first]};
      auto const second{marksOfOneColor[winnerPair.second]};
      marksOfOneColor.clear();
      marksOfOneColor.push_back(first);
      marksOfOneColor.push_back(second);
    }
  };

  double const redDistance = cv::norm(markPos.row(0) - markPos.row(1));
  double const greenDistance = cv::norm(markPos.row(2) - markPos.row(3));
  double const blueDistance = cv::norm(markPos.row(4) - markPos.row(5));
  filterSingleColor(marks.red, redDistance);
  filterSingleColor(marks.green, greenDistance);
  filterSingleColor(marks.blue, blueDistance);
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
