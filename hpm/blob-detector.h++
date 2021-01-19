#pragma once

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

#include <hpm/detection-result.h++>
#include <hpm/simple-types.h++>

hpm::DetectionResult blobDetect(cv::InputArray image,
                                bool showIntermediateImages);

hpm::DetectionResult blobDetect(cv::InputArray image);

hpm::CameraFramedPosition blobToPosition(hpm::KeyPoint const &blob,
                                         double focalLength,
                                         hpm::PixelPosition const &imageCenter,
                                         double markerDiameter);
