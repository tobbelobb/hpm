#pragma once

#include <hpm/marks.h++>
#include <hpm/simple-types.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

hpm::Marks blobDetect(cv::InputArray image, bool showIntermediateImages);

hpm::Marks blobDetect(cv::InputArray image);

hpm::CameraFramedPosition blobToPosition(hpm::KeyPoint const &blob,
                                         double focalLength,
                                         hpm::PixelPosition const &imageCenter,
                                         double markerDiameter);
