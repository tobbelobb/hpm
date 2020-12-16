#pragma once

#include <opencv2/core.hpp>

#include <hpm/types.h++>

hpm::DetectionResult blobDetect(cv::InputArray image,
                                hpm::ColorBounds const &colorBounds);

hpm::DetectionResult blobDetect(cv::InputArray image);

hpm::CameraFramedPosition blobToPosition(hpm::KeyPoint const &blob,
                                         double focalLength,
                                         hpm::PixelPosition const &imageCenter,
                                         double markerDiameter);
