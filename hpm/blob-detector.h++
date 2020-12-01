#pragma once

#include <opencv2/core.hpp>

#include <hpm/types.h++>

DetectionResult blobDetect(cv::InputArray image);

CameraFramedPosition blobToPosition(cv::KeyPoint const &blob,
                                    double focalLength,
                                    cv::Point2f const &imageCenter,
                                    double markerDiameter);
