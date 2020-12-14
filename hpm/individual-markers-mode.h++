#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

#include <hpm/blob-detector.h++>
#include <hpm/types.h++>

// function : findMarks
//
// description : Finds individual markers in image.
//               Groups them in red, green, and blue markers.
//               Each group is unsorted.
DetectionResult findMarks(cv::InputArray undistortedImage);

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
// Also, results come out unsorted for now
std::vector<CameraFramedPosition> findIndividualMarkerPositions(
    cv::InputArray undistortedImage, double knownMarkerDiameter,
    double focalLength, PixelPosition const &imageCenter,
    bool showIntermediateImages, bool showResultImage);
