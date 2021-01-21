#include <iostream>
#include <numeric>
#include <vector>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
ENABLE_WARNINGS

#include <boost/ut.hpp> //import boost.ut;

#include <hpm/hpm.h++>
#include <hpm/identified-hp-marks.h++>
#include <hpm/individual-markers-mode.h++>
#include <hpm/solve-pnp.h++>
#include <hpm/test-util.h++> // getPath

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;
  // clang-format off
  cv::Mat const openScadCameraMatrix = (cv::Mat_<double>(3, 3) << 3377.17,    0.00, 1280.0,
                                                                     0.00, 3378.36,  671.5,
                                                                     0.00,    0.00,    1.0);
  // clang-format on

  "Mover pose from OpenScad generated image"_test = [&openScadCameraMatrix] {
    std::string const imageFileName{hpm::getPath(
        "test-images/"
        "generated_benchmark_nr6_32_elevated_150p43_0_0_0_30_0_0_1500.png")};
    cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
    expect((not image.empty()) >> fatal);

    auto const &cameraMatrix = openScadCameraMatrix;
    SixDof const cameraWorldPose{{2.61799387799149, 0.0, 0.0},
                                 {0, -750.0, 1299.03810567666}};
    ProvidedMarkerPositions const providedMarkerPositions{
        -72.4478, -125.483, 150.43, 72.4478,   -125.483, 150.43,
        144.8960, 0.0,      150.43, 72.4478,   125.483,  150.43,
        -72.4478, 125.483,  150.43, -144.8960, 0.0,      150.43};
    DetectionResult marks{findMarks(image)};
    double const meanFocalLength{std::midpoint(cameraMatrix.at<double>(0, 0),
                                               cameraMatrix.at<double>(1, 1))};
    PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                    cameraMatrix.at<double>(1, 2)};
    filterMarksByDistance(marks, providedMarkerPositions, meanFocalLength,
                          imageCenter, 32.0);
    IdentifiedHpMarks const identifiedMarks{marks};

    expect((identifiedMarks.allIdentified()) >> fatal);

    std::optional<SixDof> const effectorPoseRelativeToCamera{
        solvePnp(cameraMatrix, providedMarkerPositions, identifiedMarks)};

    expect((effectorPoseRelativeToCamera.has_value()) >> fatal);

    SixDof const pose{effectorWorldPose(effectorPoseRelativeToCamera.value(),
                                        cameraWorldPose)};

    auto constexpr EPS{0.34_d};
    expect(abs(pose.x()) < EPS) << "translation X";
    expect(abs(pose.y()) < EPS) << "translation Y";
    expect(abs(pose.z()) < EPS) << "translation Z";
    expect(abs(pose.rotX()) < EPS) << "rotation X";
    expect(abs(pose.rotY()) < EPS) << "rotation Y";
    expect(abs(pose.rotZ()) < EPS) << "rotation Z";
  };

  return 0;
}
