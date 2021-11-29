#include <hpm/hpm.h++>

#include <hpm/warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/calib3d.hpp> // Rodrigues
ENABLE_WARNINGS

using namespace hpm;

auto effectorWorldPose(SixDof const &effectorPoseRelativeToCamera,
                       SixDof const &cameraPoseRelativeToWorld) -> SixDof {

  auto const rotationMatrix = [&]() {
    cv::Matx33d rotationMatrix_;
    cv::Rodrigues(-cameraPoseRelativeToWorld.rotation, rotationMatrix_);
    return rotationMatrix_;
  }();

  auto rotation = effectorPoseRelativeToCamera.rotation -
                  cameraPoseRelativeToWorld.rotation;
  for (int i{0}; i < rotation.rows; ++i) {
    rotation(i) = remainder(rotation(i), CV_PI);
  }

  const auto translation =
      rotationMatrix * effectorPoseRelativeToCamera.translation +
      cameraPoseRelativeToWorld.translation;
  const auto reprojectionError = effectorPoseRelativeToCamera.reprojectionError;
  return {rotation, translation, reprojectionError};
}

auto effectorPoseRelativeToBed(SixDof const &effectorPoseRelativeToCamera,
                               SixDof const &bedPoseRelativeToCamera)
    -> SixDof {
  // std::cout << "effectorPoseRelativeToCamera=\n" <<
  // effectorPoseRelativeToCamera << '\n'; std::cout <<
  // "bedPoseRelativeToCamera=\n" << bedPoseRelativeToCamera << '\n';

  auto const rotationMatrix = [&]() {
    cv::Matx33d rotationMatrix_;
    cv::Rodrigues(-bedPoseRelativeToCamera.rotation, rotationMatrix_);
    return rotationMatrix_;
  }();

  auto rotation =
      effectorPoseRelativeToCamera.rotation - bedPoseRelativeToCamera.rotation;
  for (int i{0}; i < rotation.rows; ++i) {
    rotation(i) = remainder(rotation(i), CV_PI);
  }

  const auto translation =
      rotationMatrix * (effectorPoseRelativeToCamera.translation -
                        bedPoseRelativeToCamera.translation);

  // std::cout << "new translation=\n" << translation << '\n';

  const auto reprojectionError =
      std::max(effectorPoseRelativeToCamera.reprojectionError,
               bedPoseRelativeToCamera.reprojectionError);
  return {rotation, translation, reprojectionError};
}
