#pragma once

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

#include <hpm/detection-result.h++>
#include <hpm/simple-types.h++>

hpm::DetectionResult ellipseDetect(cv::InputArray image,
                                   bool showIntermediateImages);

hpm::DetectionResult ellipseDetect(cv::InputArray image);
