#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include <hpm/types.h++>

hpm::SixDof effectorWorldPose(hpm::SixDof const &effectorPoseRelativeToCamera,
                              hpm::SixDof const &cameraPoseRelativeToWorld);
