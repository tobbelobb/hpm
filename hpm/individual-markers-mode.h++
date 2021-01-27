#pragma once

#include <hpm/blob-detector.h++>
#include <hpm/detection-result.h++>
#include <hpm/identified-hp-marks.h++>
#include <hpm/simple-types.h++>

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>

std::pair<hpm::IdentifiedHpMarks, hpm::DetectionResult>
find(cv::InputArray undistortedImage,
     hpm::ProvidedMarkerPositions const &markPos, double const focalLength,
     hpm::PixelPosition const &imageCenter, double const markerDiameter,
     bool showIntermediateImages = false, bool verbose = false);

hpm::DetectionResult
findMarks(cv::InputArray undistortedImage,
          hpm::ProvidedMarkerPositions const &markPos, double const focalLength,
          hpm::PixelPosition const &imageCenter, double const markerDiameter,
          bool showIntermediateImages = false, bool verbose = false);

std::vector<hpm::CameraFramedPosition>
findIndividualMarkerPositions(hpm::DetectionResult const &detectionResult,
                              double knownMarkerDiameter, double focalLength,
                              hpm::PixelPosition const &imageCenter);
