#pragma once

#include <hpm/identified-marks.h++>
#include <hpm/marks.h++>
#include <hpm/simple-types.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

#include <iostream>
#include <vector>

namespace hpm {
struct FindResult {
  IdentifiedMarks identifiedMarks;
  Marks marks;
};
} // namespace hpm

hpm::FindResult find(cv::InputArray undistortedImage,
                     hpm::ProvidedMarkerPositions const &markPos,
                     double const focalLength,
                     hpm::PixelPosition const &imageCenter,
                     double const markerDiameter,
                     bool showIntermediateImages = false, bool verbose = false,
                     bool fitByDistance = false);

hpm::Marks findMarks(cv::InputArray undistortedImage,
                     hpm::ProvidedMarkerPositions const &markPos,
                     double const focalLength,
                     hpm::PixelPosition const &imageCenter,
                     double const markerDiameter,
                     bool showIntermediateImages = false, bool verbose = false,
                     bool fitByDistance = false);

std::vector<hpm::CameraFramedPosition>
findIndividualMarkerPositions(hpm::Marks const &marks,
                              double knownMarkerDiameter, double focalLength,
                              hpm::PixelPosition const &imageCenter);
