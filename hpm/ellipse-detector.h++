#pragma once

#include <hpm/marks.h++>
#include <hpm/simple-types.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

hpm::Marks ellipseDetect(cv::InputArray image, bool showIntermediateImages);

hpm::Marks ellipseDetect(cv::InputArray image);

hpm::CameraFramedPosition
ellipseToPosition(hpm::KeyPoint const &ellipse, double focalLength,
                  hpm::PixelPosition const &imageCenter, double markerDiameter);
