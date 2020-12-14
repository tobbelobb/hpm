#pragma once

#include <opencv2/core.hpp>

#include <hpm/types.h++>

DetectionResult blobDetect(cv::InputArray image);

CameraFramedPosition blobToPosition(hpm::KeyPoint const &blob,
                                    double focalLength,
                                    PixelPosition const &imageCenter,
                                    double markerDiameter);
