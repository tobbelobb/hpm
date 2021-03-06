#include <hpm/find.h++>
#include <hpm/hpm.h++>
#include <hpm/solve-pnp.h++>
#include <hpm/test-util.h++> // getPath

#include <hpm/warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
ENABLE_WARNINGS

#include <boost/ut.hpp> //import boost.ut;

#include <iostream>
#include <numeric>
#include <vector>

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;
  // clang-format off
  cv::Mat const openScadCameraMatrix = (cv::Mat_<double>(3, 3) << 3375.85,    0.00, 1280.0,
                                                                     0.00, 3375.85,  671.5,
                                                                     0.00,    0.00,    1.0);
  cv::Mat const openScadCameraMatrix2x = (cv::Mat_<double>(3, 3) << 3375.85*2.0,    0.00*2.0, 1280.0*2.0,
                                                                       0.00*2.0, 3375.85*2.0,  671.5*2.0,
                                                                       0.00*2.0,    0.00*2.0,    1.0);
  // clang-format on

  "Mover pose from OpenScad generated image white spheres"_test =
      [&openScadCameraMatrix] {
        double constexpr markerDiameter{32.0};
        std::string const imageFileName{hpm::getPath(
            "test-images/"
            "generated_benchmark_nr6_32_elevated_150p43_0_0_0_30_0_0_"
            "1500_white.png")};
        cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
        expect((not image.empty()) >> fatal);

        auto const &cameraMatrix = openScadCameraMatrix;
        SixDof const cameraWorldPose{{2.61799387799149, 0.0, 0.0},
                                     {0, -750.0, 1299.03810567666}};
        ProvidedMarkerPositions const providedMarkerPositions{
            -72.4478, -125.483, 150.43, 72.4478,  -125.483, 150.43,
            146.895,  -3.4642,  150.43, 64.446,   139.34,   150.43,
            -68.4476, 132.411,  150.43, -160.895, -27.7129, 150.43};
        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                        cameraMatrix.at<double>(1, 2)};

        MarkerParams const markerParams{providedMarkerPositions,
                                        markerDiameter};

        FinderImage const finderImage{image, meanFocalLength, imageCenter};

        auto const marks{findMarks(finderImage, markerParams,
                                   {.m_showIntermediateImages = false,
                                    .m_verbose = false,
                                    .m_fitByDistance = true})};

        SolvePnpPoints const points{marks, markerDiameter, meanFocalLength,
                                    imageCenter, MarkerType::SPHERE};

        expect((points.allIdentified()) >> fatal);

        std::optional<SixDof> const effectorPoseRelativeToCamera{
            solvePnp(cameraMatrix, providedMarkerPositions, points)};

        expect((effectorPoseRelativeToCamera.has_value()) >> fatal);

        SixDof const pose{effectorWorldPose(
            effectorPoseRelativeToCamera.value(), cameraWorldPose)};

        auto constexpr EPS{0.11_d};
        expect(abs(pose.x()) < EPS) << "translation X";
        expect(abs(pose.y()) < EPS) << "translation Y";
        expect(abs(pose.z()) < EPS) << "translation Z";
        expect(abs(pose.rotX()) < EPS) << "rotation X";
        expect(abs(pose.rotY()) < EPS) << "rotation Y";
        expect(abs(pose.rotZ()) < EPS) << "rotation Z";
      };

  "Mover pose from OpenScad generated image white disks"_test =
      [&openScadCameraMatrix2x] {
        double constexpr markerDiameter{70.0};
        std::string const imageFileName{hpm::getPath(
            "test-images/"
            "full_tilted_pose_0_0_0_30_0_0_1500_white_disks_doubled.png")};
        cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
        expect((not image.empty()) >> fatal);

        auto const &cameraMatrix = openScadCameraMatrix2x;
        SixDof const cameraWorldPose{{2.61799387799149, 0.0, 0.0},
                                     {0, -750.0, 1299.03810567666}};
        ProvidedMarkerPositions const providedMarkerPositions{
            -72.4478, -125.483, 136.03, 72.4478,  -125.483, 136.03,
            146.895,  -3.4642,  136.03, 64.446,   139.34,   136.03,
            -68.4476, 132.411,  136.03, -160.895, -27.7129, 136.03};

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                        cameraMatrix.at<double>(1, 2)};

        MarkerParams const markerParams{providedMarkerPositions,
                                        markerDiameter};

        FinderImage const finderImage{image, meanFocalLength, imageCenter};

        auto const marks{findMarks(finderImage, markerParams,
                                   {.m_showIntermediateImages = false,
                                    .m_verbose = false,
                                    .m_fitByDistance = true})};

        SolvePnpPoints const points{
            marks,       markerDiameter,   meanFocalLength,
            imageCenter, MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};
        expect((points.allIdentified()) >> fatal);

        std::optional<SixDof> const effectorPoseRelativeToCamera{
            solvePnp(cameraMatrix, providedMarkerPositions, points)};

        expect((effectorPoseRelativeToCamera.has_value()) >> fatal);

        SixDof const pose{effectorWorldPose(
            effectorPoseRelativeToCamera.value(), cameraWorldPose)};

        auto constexpr EPS{0.023_d};
        auto constexpr EPSZ{0.34_d};
        expect(abs(pose.x()) < EPS) << "translation X";
        expect(abs(pose.y()) < EPS) << "translation Y";
        expect(abs(pose.z()) < EPSZ) << "translation Z";
        expect(abs(pose.rotX()) < EPS) << "rotation X";
        expect(abs(pose.rotY()) < EPS) << "rotation Y";
        expect(abs(pose.rotZ()) < EPS) << "rotation Z";
      };

  return 0;
}
