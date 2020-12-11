#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include <hpm/types.h++>

SixDof effectorWorldPose(SixDof const &effectorPoseRelativeToCamera,
                         SixDof const &cameraPoseRelativeToWorld);
