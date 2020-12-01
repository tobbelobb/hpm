#pragma once

#include <optional>

#include <opencv2/core.hpp>

#include <hpm/types.h++>

/* We want to use OpenCV's implementation of an IPPE PnP solver
 * as long as the points are co-planar.
 * If points are not co-planar, then we want to use OpenCV's
 * coming implementation of the SQPnP algorithm.
 */

std::optional<SixDof> solvePnp(cv::InputArray cameraMatrix,
                               cv::InputArray markerPositionsRelativeToNozzle,
                               IdentifiedHpMarks const &marks);
