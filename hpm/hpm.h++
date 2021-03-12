#pragma once

#include <hpm/six-dof.h++>

#include <hpm/warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

#include <vector>

hpm::SixDof effectorWorldPose(hpm::SixDof const &effectorPoseRelativeToCamera,
                              hpm::SixDof const &cameraPoseRelativeToWorld);
