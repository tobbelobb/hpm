#pragma once

#include <hpm/marks.h++>
#include <hpm/simple-types.h++>
#include <hpm/solve-pnp.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

#include <iostream>
#include <vector>

hpm::Marks findMarks(cv::InputArray undistortedImage,
                     hpm::ProvidedMarkerPositions const &markPos,
                     double const focalLength,
                     hpm::PixelPosition const &imageCenter,
                     double const markerDiameter,
                     bool showIntermediateImages = false, bool verbose = false,
                     bool fitByDistance = false,
                     hpm::PixelPosition const &expectedTopLeftestCenter =
                         hpm::PixelPosition(0.0, 0.0));

std::vector<hpm::CameraFramedPosition>
findIndividualMarkerPositions(hpm::Marks const &marks,
                              double knownMarkerDiameter, double focalLength,
                              hpm::PixelPosition const &imageCenter);
