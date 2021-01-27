#pragma once

#include <hpm/marks.h++>
#include <hpm/simple-types.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

hpm::Marks ellipseDetect(cv::InputArray image,
                         bool showIntermediateImages = false);
