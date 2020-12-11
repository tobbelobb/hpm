#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/calib3d.hpp> // Rodrigues
#pragma GCC diagnostic pop

#include <hpm/hpm.h++>

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
