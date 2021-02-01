#include <hpm/blob-detector.h++>
#include <hpm/ellipse-detector.h++>
#include <hpm/find.h++>
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
          bool showIntermediateImages, bool verbose, bool filterByDistance)
    -> std::pair<hpm::IdentifiedMarks, hpm::Marks> {
  Marks const marks{findMarks(
      undistortedImage, markPos, focalLength, imageCenter, markerDiameter,
      showIntermediateImages, verbose, filterByDistance)};
  IdentifiedMarks const identifiedMarks{marks, markerDiameter / 2.0,
                                        focalLength, imageCenter};
  return {identifiedMarks, marks};
}

auto findMarks(cv::InputArray undistortedImage,
               hpm::ProvidedMarkerPositions const &markPos,
               double const focalLength, PixelPosition const &imageCenter,
               double const markerDiameter, bool showIntermediateImages,
               bool verbose, bool filterByDistance) -> Marks {

  auto marks{ellipseDetect(undistortedImage, showIntermediateImages)};

  if (showIntermediateImages) {
    showImage(imageWith(undistortedImage, marks),
              "found-marks-before-filtering-by-distance.png");
  }
  if (filterByDistance) {
    marks.filterAndSortByDistance(markPos, focalLength, imageCenter,
                                  markerDiameter);
    if (showIntermediateImages) {
      showImage(imageWith(undistortedImage, marks),
                "found-marks-after-filtering-by-distance.png");
    }
  }
  if (verbose) {
    std::cout << "Found " << marks.red.size() << " red markers, "
              << marks.green.size() << " green markers, and "
              << marks.blue.size() << " blue markers\n";
  }

  return marks;
}

auto findIndividualMarkerPositions(Marks const &marks,
                                   double const knownMarkerDiameter,
                                   double const focalLength,
                                   PixelPosition const &imageCenter)
    -> std::vector<CameraFramedPosition> {
  std::vector<CameraFramedPosition> positions{};
  positions.reserve(marks.size());
  for (auto const &detected : marks.red) {
    positions.emplace_back(
        detected.toPosition(focalLength, imageCenter, knownMarkerDiameter));
  }
  for (auto const &detected : marks.green) {
    positions.emplace_back(
        detected.toPosition(focalLength, imageCenter, knownMarkerDiameter));
  }
  for (auto const &detected : marks.blue) {
    positions.emplace_back(
        detected.toPosition(focalLength, imageCenter, knownMarkerDiameter));
  }

  return positions;
}
