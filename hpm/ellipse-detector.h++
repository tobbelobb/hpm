#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/core.hpp>
#pragma GCC diagnostic pop

#include <hpm/detection-result.h++>
#include <hpm/simple-types.h++>

hpm::DetectionResult ellipseDetect(cv::InputArray image,
                                   bool showIntermediateImages);

hpm::DetectionResult ellipseDetect(cv::InputArray image);
