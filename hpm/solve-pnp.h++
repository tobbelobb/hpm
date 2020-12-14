#pragma once

#include <optional>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/core.hpp>
#pragma GCC diagnostic pop

#include <hpm/types.h++>

/* We want to use OpenCV's implementation of an IPPE PnP solver
 * as long as the points are co-planar.
 * If points are not co-planar, then we want to use OpenCV's
 * coming implementation of the SQPnP algorithm.
 */

std::optional<hpm::SixDof>
solvePnp(cv::InputArray cameraMatrix,
         cv::InputArray markerPositionsRelativeToNozzle,
         hpm::IdentifiedHpMarks const &marks);
