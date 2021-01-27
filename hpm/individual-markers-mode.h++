#pragma once

#include <hpm/blob-detector.h++>
#include <hpm/detection-result.h++>
#include <hpm/simple-types.h++>

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>

// function : findMarks
//
// description : Finds individual markers in image.
//               Groups them in red, green, and blue markers.
//               Each group is unsorted.
//               The colors are defined to be any color within the
//               specified bounds.
hpm::DetectionResult findMarks(cv::InputArray undistortedImage,
                               bool showIntermediateImages);

hpm::DetectionResult findMarks(cv::InputArray undistortedImage);

// function : findIndividualMarkerPositions
//
// description : Detects markers in the image and transforms the
// pixel values into a position in the camera's coordinate system,
// using the same length unit as knownMarkerDiameter uses.
//
// focalLength and imageCenter uses pixels as their length unit
//
// The marker detection, and the interpretation of marker detection results are
// (sadly) intertwined, so find performs both detection and interpretation
//
// Also, results come out unsorted
std::vector<hpm::CameraFramedPosition>
findIndividualMarkerPositions(hpm::DetectionResult const &blobs,
                              double knownMarkerDiameter, double focalLength,
                              hpm::PixelPosition const &imageCenter);

// function : findIndividualMarkerPositions
//
// description : A detection result sometimes includes
// too many markers. In that case, we can use the knowledge
// we have about the real distances between the markers,
// to try to weed out the false markers.
void filterMarksByDistance(hpm::DetectionResult &marks,
                           hpm::ProvidedMarkerPositions const &markPos,
                           double const focalLength,
                           hpm::PixelPosition const &imageCenter,
                           double const markerDiameter);
