#pragma once

#include <vector>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

#include <hpm/six-dof.h++>

hpm::SixDof effectorWorldPose(hpm::SixDof const &effectorPoseRelativeToCamera,
                              hpm::SixDof const &cameraPoseRelativeToWorld);
