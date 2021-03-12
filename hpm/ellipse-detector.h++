#pragma once

#include <hpm/marks.h++>
#include <hpm/simple-types.h++>

#include <hpm/warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

std::vector<hpm::Ellipse> rawEllipseDetect(cv::InputArray image,
                                           bool showIntermediateImages);

std::vector<hpm::Ellipse>
ellipseDetect(cv::InputArray image, bool showIntermediateImages = false,
              hpm::PixelPosition const &expectedTopLeftestCenter =
                  hpm::PixelPosition(0.0, 0.0));
