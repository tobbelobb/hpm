#pragma once

#include <optional>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

#include <hpm/identified-hp-marks.h++>
#include <hpm/simple-types.h++>
#include <hpm/six-dof.h++>

/* We want to use OpenCV's implementation of an IPPE PnP solver
 * as long as the points are co-planar.
 * If points are not co-planar, then we want to use OpenCV's
 * coming implementation of the SQPnP algorithm.
 */

std::optional<hpm::SixDof>
solvePnp(cv::InputArray cameraMatrix,
         cv::InputArray markerPositionsRelativeToNozzle,
         hpm::IdentifiedHpMarks const &marks);
