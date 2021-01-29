#pragma once

#include <hpm/blob-detector.h++>
#include <hpm/identified-marks.h++>
#include <hpm/marks.h++>
#include <hpm/simple-types.h++>

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>

std::pair<hpm::IdentifiedMarks, hpm::Marks>
find(cv::InputArray undistortedImage,
     hpm::ProvidedMarkerPositions const &markPos, double const focalLength,
     hpm::PixelPosition const &imageCenter, double const markerDiameter,
     bool showIntermediateImages = false, bool verbose = false,
     bool noFilterByDistance = false);

hpm::Marks findMarks(cv::InputArray undistortedImage,
                     hpm::ProvidedMarkerPositions const &markPos,
                     double const focalLength,
                     hpm::PixelPosition const &imageCenter,
                     double const markerDiameter,
                     bool showIntermediateImages = false, bool verbose = false,
                     bool noFilterByDistance = false);

std::vector<hpm::CameraFramedPosition>
findIndividualMarkerPositions(hpm::Marks const &marks,
                              double knownMarkerDiameter, double focalLength,
                              hpm::PixelPosition const &imageCenter);
