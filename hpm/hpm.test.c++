#include <hpm/find.h++>
#include <hpm/hpm.h++>
#include <hpm/solve-pnp.h++>
#include <hpm/test-util.h++> // getPath
#include <hpm/util.h++>

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

  "Effector pose from OpenScad generated image white spheres"_test =
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

  "Effector pose from OpenScad generated image white disks"_test =
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

  "Effector pose relative to bed pose from OpenScad generated image white disks"_test =
      [&openScadCameraMatrix2x] {
        double constexpr markerDiameter{90.0};
        std::string const imageFileName{
            hpm::getPath("test-images/"
                         "bed_markers_test_0_0_0_30_0_0_2000_doubled.png")};
        cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
        expect((not image.empty()) >> fatal);

        auto const &cameraMatrix = openScadCameraMatrix2x;

        ProvidedMarkerPositions const effectorMarkerPositions{
            -72.4478, -125.483, 136.03, 72.4478,  -125.483, 136.03,
            146.895,  -3.4642,  136.03, 64.446,   139.34,   136.03,
            -68.4476, 132.411,  136.03, -160.895, -27.7129, 136.03};

        ProvidedMarkerPositions const bedMarkerPositions{
            -300.0, -300.0, 0.0, 250.0, -300.0, 0.0, 250.0,  0.0,   0.0,
            230.0,  380.0,  0.0, 0.0,   400.0,  0.0, -300.0, 380.0, 0.0};

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                        cameraMatrix.at<double>(1, 2)};

        MarkerParams const effectorMarkerParams{effectorMarkerPositions,
                                                markerDiameter};
        MarkerParams const bedMarkerParams{bedMarkerPositions, markerDiameter};

        FinderImage const finderImage{image, meanFocalLength, imageCenter};

        auto const effectorMarks{findMarks(finderImage, effectorMarkerParams,
                                           {.m_showIntermediateImages = false,
                                            .m_verbose = false,
                                            .m_fitByDistance = true})};

        auto const bedMarks{findMarks(finderImage, bedMarkerParams,
                                      {.m_showIntermediateImages = false,
                                       .m_verbose = false,
                                       .m_fitByDistance = true},
                                      {0, 0, 0}, false, effectorMarks)};

        SolvePnpPoints const effectorPoints{
            effectorMarks, markerDiameter,   meanFocalLength,
            imageCenter,   MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        SolvePnpPoints const bedPoints{
            bedMarks,    markerDiameter,   meanFocalLength,
            imageCenter, MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        expect((effectorPoints.allIdentified()) >> fatal);
        expect((bedPoints.allIdentified()) >> fatal);

        std::optional<SixDof> const effectorPoseRelativeToCamera{
            solvePnp(cameraMatrix, effectorMarkerPositions, effectorPoints)};

        std::optional<SixDof> const bedPoseRelativeToCamera{
            solvePnp(cameraMatrix, bedMarkerPositions, bedPoints)};

        expect((effectorPoseRelativeToCamera.has_value()) >> fatal);
        expect((bedPoseRelativeToCamera.has_value()) >> fatal);

        SixDof const pose{
            effectorPoseRelativeToBed(effectorPoseRelativeToCamera.value(),
                                      bedPoseRelativeToCamera.value())};

        auto constexpr EPSX{0.023_d};
        auto constexpr EPSY{0.079_d};
        auto constexpr EPSZ{0.34_d};
        auto constexpr EPSROT{0.023_d};
        expect(abs(pose.x()) < EPSX) << "translation X";
        expect(abs(pose.y()) < EPSY) << "translation Y";
        expect(abs(pose.z() - 10.0) < EPSZ) << "translation Z";
        expect(abs(pose.rotX()) < EPSROT) << "rotation X";
        expect(abs(pose.rotY()) < EPSROT) << "rotation Y";
        expect(abs(pose.rotZ()) < EPSROT) << "rotation Z";
      };

  "Effector pose relative to bed pose from OpenScad generated image white disks"_test =
      [&openScadCameraMatrix2x] {
        double constexpr markerDiameter{90.0};
        std::string const imageFileName{hpm::getPath(
            "test-images/"
            "bed_markers_test_0_0_0_30_0_0_2500_doubled_10_deg_twist.png")};
        cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
        expect((not image.empty()) >> fatal);

        auto const &cameraMatrix = openScadCameraMatrix2x;

        ProvidedMarkerPositions const effectorMarkerPositions{
            -72.4478, -125.483, 136.03, 72.4478,  -125.483, 136.03,
            146.895,  -3.4642,  136.03, 64.446,   139.34,   136.03,
            -68.4476, 132.411,  136.03, -160.895, -27.7129, 136.03};

        ProvidedMarkerPositions const bedMarkerPositions{
            -300.0, -300.0, 0.0, 250.0, -300.0, 0.0, 250.0,  0.0,   0.0,
            230.0,  380.0,  0.0, 0.0,   400.0,  0.0, -300.0, 380.0, 0.0};

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                        cameraMatrix.at<double>(1, 2)};

        MarkerParams const effectorMarkerParams{effectorMarkerPositions,
                                                markerDiameter};
        MarkerParams const bedMarkerParams{bedMarkerPositions, markerDiameter};

        FinderImage const finderImage{image, meanFocalLength, imageCenter};

        auto const effectorMarks{findMarks(finderImage, effectorMarkerParams,
                                           {.m_showIntermediateImages = false,
                                            .m_verbose = false,
                                            .m_fitByDistance = true})};

        auto const bedMarks{findMarks(finderImage, bedMarkerParams,
                                      {.m_showIntermediateImages = false,
                                       .m_verbose = false,
                                       .m_fitByDistance = true},
                                      {0, 0, -1}, false, effectorMarks)};

        SolvePnpPoints const effectorPoints{
            effectorMarks, markerDiameter,   meanFocalLength,
            imageCenter,   MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        SolvePnpPoints const bedPoints{
            bedMarks,    markerDiameter,   meanFocalLength,
            imageCenter, MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        expect((effectorPoints.allIdentified()) >> fatal);
        expect((bedPoints.allIdentified()) >> fatal);

        std::optional<SixDof> const effectorPoseRelativeToCamera{
            solvePnp(cameraMatrix, effectorMarkerPositions, effectorPoints)};

        std::optional<SixDof> const bedPoseRelativeToCamera{
            solvePnp(cameraMatrix, bedMarkerPositions, bedPoints)};

        expect((effectorPoseRelativeToCamera.has_value()) >> fatal);
        expect((bedPoseRelativeToCamera.has_value()) >> fatal);

        SixDof const pose{
            effectorPoseRelativeToBed(effectorPoseRelativeToCamera.value(),
                                      bedPoseRelativeToCamera.value())};

        auto constexpr EPSX{0.11_d};
        auto constexpr EPSY{0.12_d};
        auto constexpr EPSZ{0.34_d};
        auto constexpr EPSROT{0.023_d};
        expect(abs(pose.x() - 20.0) < EPSX) << "translation X";
        expect(abs(pose.y() - 60.0) < EPSY) << "translation Y";
        expect(abs(pose.z() - 10.0) < EPSZ) << "translation Z";
        expect(abs(pose.rotX()) < EPSROT) << "rotation X";
        expect(abs(pose.rotY()) < EPSROT) << "rotation Y";
        expect(abs(pose.rotZ()) < EPSROT) << "rotation Z";
      };

  "Effector pose high up relative to bed pose from OpenScad generated image white disks"_test =
      [&openScadCameraMatrix2x] {
        double constexpr markerDiameter{90.0};
        std::string const imageFileName{hpm::getPath(
            "test-images/"
            "bed_markers_test_0_0_0_30_0_0_5500_doubled_mover_1000.png")};
        cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
        expect((not image.empty()) >> fatal);

        auto const &cameraMatrix = openScadCameraMatrix2x;

        ProvidedMarkerPositions const effectorMarkerPositions{
            -72.4478, -125.483, 136.03, 72.4478,  -125.483, 136.03,
            146.895,  -3.4642,  136.03, 64.446,   139.34,   136.03,
            -68.4476, 132.411,  136.03, -160.895, -27.7129, 136.03};

        ProvidedMarkerPositions const bedMarkerPositions{
            -300.0, -300.0, 0.0, 250.0, -300.0, 0.0, 250.0,  0.0,   0.0,
            230.0,  380.0,  0.0, 0.0,   400.0,  0.0, -300.0, 380.0, 0.0};

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                        cameraMatrix.at<double>(1, 2)};

        MarkerParams const effectorMarkerParams{effectorMarkerPositions,
                                                markerDiameter};
        MarkerParams const bedMarkerParams{bedMarkerPositions, markerDiameter};

        FinderImage const finderImage{image, meanFocalLength, imageCenter};

        auto const effectorMarks{findMarks(finderImage, effectorMarkerParams,
                                           {.m_showIntermediateImages = false,
                                            .m_verbose = false,
                                            .m_fitByDistance = true})};

        auto const bedMarks{findMarks(finderImage, bedMarkerParams,
                                      {.m_showIntermediateImages = false,
                                       .m_verbose = false,
                                       .m_fitByDistance = true},
                                      {0, 0, -1}, false, effectorMarks)};

        SolvePnpPoints const effectorPoints{
            effectorMarks, markerDiameter,   meanFocalLength,
            imageCenter,   MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        SolvePnpPoints const bedPoints{
            bedMarks,    markerDiameter,   meanFocalLength,
            imageCenter, MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        expect((effectorPoints.allIdentified()) >> fatal);
        expect((bedPoints.allIdentified()) >> fatal);

        std::optional<SixDof> const effectorPoseRelativeToCamera{
            solvePnp(cameraMatrix, effectorMarkerPositions, effectorPoints)};

        std::optional<SixDof> const bedPoseRelativeToCamera{
            solvePnp(cameraMatrix, bedMarkerPositions, bedPoints)};

        expect((effectorPoseRelativeToCamera.has_value()) >> fatal);
        expect((bedPoseRelativeToCamera.has_value()) >> fatal);

        SixDof const pose{
            effectorPoseRelativeToBed(effectorPoseRelativeToCamera.value(),
                                      bedPoseRelativeToCamera.value())};

        auto constexpr EPSX{0.11_d};
        auto constexpr EPSY{0.12_d};
        auto constexpr EPSZ{0.34_d};
        auto constexpr EPSROT{0.023_d};
        expect(abs(pose.x()) < EPSX) << "translation X";
        expect(abs(pose.y()) < EPSY) << "translation Y";
        expect(abs(pose.z() - 1000.0) < EPSZ) << "translation Z";
        expect(abs(pose.rotX()) < EPSROT) << "rotation X";
        expect(abs(pose.rotY()) < EPSROT) << "rotation Y";
        expect(abs(pose.rotZ()) < EPSROT) << "rotation Z";
      };

  "Effector pose relative to twisted bed pose from OpenScad generated image white disks"_test =
      [&openScadCameraMatrix2x] {
        double constexpr markerDiameter{90.0};
        std::string const imageFileName{hpm::getPath(
            "test-images/"
            "bed_markers_test_0_0_0_30_0_0_2500_doubled_10_deg_twist_different_ways.png")};
        cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
        expect((not image.empty()) >> fatal);

        auto const &cameraMatrix = openScadCameraMatrix2x;

        ProvidedMarkerPositions const effectorMarkerPositions{
            -72.4478, -125.483, 136.03, 72.4478,  -125.483, 136.03,
            146.895,  -3.4642,  136.03, 64.446,   139.34,   136.03,
            -68.4476, 132.411,  136.03, -160.895, -27.7129, 136.03};

        ProvidedMarkerPositions const bedMarkerPositions{
            -300.0, -300.0, 0.0, 250.0, -300.0, 0.0, 250.0,  0.0,   0.0,
            230.0,  380.0,  0.0, 0.0,   400.0,  0.0, -300.0, 380.0, 0.0};

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                        cameraMatrix.at<double>(1, 2)};

        MarkerParams const effectorMarkerParams{effectorMarkerPositions,
                                                markerDiameter};
        MarkerParams const bedMarkerParams{bedMarkerPositions, markerDiameter};

        FinderImage const finderImage{image, meanFocalLength, imageCenter};

        auto const effectorMarks{findMarks(finderImage, effectorMarkerParams,
                                           {.m_showIntermediateImages = false,
                                            .m_verbose = false,
                                            .m_fitByDistance = true})};

        auto const bedMarks{findMarks(finderImage, bedMarkerParams,
                                      {.m_showIntermediateImages = false,
                                       .m_verbose = false,
                                       .m_fitByDistance = true},
                                      {0, 0, -1}, false, effectorMarks)};

        SolvePnpPoints const effectorPoints{
            effectorMarks, markerDiameter,   meanFocalLength,
            imageCenter,   MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        SolvePnpPoints const bedPoints{
            bedMarks,    markerDiameter,   meanFocalLength,
            imageCenter, MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        expect((effectorPoints.allIdentified()) >> fatal);
        expect((bedPoints.allIdentified()) >> fatal);

        std::optional<SixDof> const effectorPoseRelativeToCamera{
            solvePnp(cameraMatrix, effectorMarkerPositions, effectorPoints)};

        std::optional<SixDof> const bedPoseRelativeToCamera{
            solvePnp(cameraMatrix, bedMarkerPositions, bedPoints)};

        expect((effectorPoseRelativeToCamera.has_value()) >> fatal);
        expect((bedPoseRelativeToCamera.has_value()) >> fatal);

        SixDof const pose{
            effectorPoseRelativeToBed(effectorPoseRelativeToCamera.value(),
                                      bedPoseRelativeToCamera.value())};

        auto constexpr EPSX{0.11_d};
        auto constexpr EPSY{0.12_d};
        auto constexpr EPSZ{0.34_d};
        auto constexpr EPSROT{0.01_d};
        auto constexpr EXPECTED_ROT_Z{-20.0 * CV_PI/180.0};
        expect(abs(pose.x() - 20.0) < EPSX) << "translation X";
        expect(abs(pose.y() - 60.0) < EPSY) << "translation Y";
        expect(abs(pose.z() - 10.0) < EPSZ) << "translation Z";
        expect(abs(pose.rotX()) < EPSROT) << "rotation X";
        expect(abs(pose.rotY()) < EPSROT) << "rotation Y";
        expect(abs(pose.rotZ() - EXPECTED_ROT_Z) < EPSROT) << "rotation Z";
      };

  "X-Tilted effector pose relative to bed pose from OpenScad generated image white disks"_test =
      [&openScadCameraMatrix2x] {
        double constexpr markerDiameter{90.0};
        std::string const imageFileName{hpm::getPath(
            "test-images/"
            "bed_markers_test_0_0_0_30_0_0_2500_doubled_mover_tilted_20_towards_camera.png")};
        cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
        expect((not image.empty()) >> fatal);

        auto const &cameraMatrix = openScadCameraMatrix2x;

        ProvidedMarkerPositions const effectorMarkerPositions{
            -72.4478, -125.483, 136.03, 72.4478,  -125.483, 136.03,
            146.895,  -3.4642,  136.03, 64.446,   139.34,   136.03,
            -68.4476, 132.411,  136.03, -160.895, -27.7129, 136.03};

        ProvidedMarkerPositions const bedMarkerPositions{
            -300.0, -300.0, 0.0, 250.0, -300.0, 0.0, 250.0,  0.0,   0.0,
            230.0,  380.0,  0.0, 0.0,   400.0,  0.0, -300.0, 380.0, 0.0};

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                        cameraMatrix.at<double>(1, 2)};

        MarkerParams const effectorMarkerParams{effectorMarkerPositions,
                                                markerDiameter};
        MarkerParams const bedMarkerParams{bedMarkerPositions, markerDiameter};

        FinderImage const finderImage{image, meanFocalLength, imageCenter};

        auto const effectorMarks{findMarks(finderImage, effectorMarkerParams,
                                           {.m_showIntermediateImages = false,
                                            .m_verbose = false,
                                            .m_fitByDistance = true})};

        auto const bedMarks{findMarks(finderImage, bedMarkerParams,
                                      {.m_showIntermediateImages = false,
                                       .m_verbose = false,
                                       .m_fitByDistance = true},
                                      {0, 0, -1}, false, effectorMarks)};

        SolvePnpPoints const effectorPoints{
            effectorMarks, markerDiameter,   meanFocalLength,
            imageCenter,   MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        SolvePnpPoints const bedPoints{
            bedMarks,    markerDiameter,   meanFocalLength,
            imageCenter, MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        expect((effectorPoints.allIdentified()) >> fatal);
        expect((bedPoints.allIdentified()) >> fatal);

        std::optional<SixDof> const effectorPoseRelativeToCamera{
            solvePnp(cameraMatrix, effectorMarkerPositions, effectorPoints)};

        std::optional<SixDof> const bedPoseRelativeToCamera{
            solvePnp(cameraMatrix, bedMarkerPositions, bedPoints)};

        expect((effectorPoseRelativeToCamera.has_value()) >> fatal);
        expect((bedPoseRelativeToCamera.has_value()) >> fatal);

        SixDof const pose{
            effectorPoseRelativeToBed(effectorPoseRelativeToCamera.value(),
                                      bedPoseRelativeToCamera.value())};

        auto constexpr EPSX{0.04_d};
        auto constexpr EPSY{0.06_d};
        auto constexpr EPSZ{0.14_d};
        auto constexpr EPSROT{0.001_d};
        auto constexpr EXPECTED_ROT_X{20.0 * CV_PI/180.0};
        expect(abs(pose.x()) < EPSX) << "translation X";
        expect(abs(pose.y()) < EPSY) << "translation Y";
        expect(abs(pose.z()) < EPSZ) << "translation Z";
        expect(abs(pose.rotX() - EXPECTED_ROT_X) < EPSROT) << "rotation X";
        expect(abs(pose.rotY()) < EPSROT) << "rotation Y";
        expect(abs(pose.rotZ()) < EPSROT) << "rotation Z";
      };

  "Y-Tilted effector pose relative to bed pose from OpenScad generated image white disks"_test =
      [&openScadCameraMatrix2x] {
        double constexpr markerDiameter{90.0};
        std::string const imageFileName{hpm::getPath(
            "test-images/"
            "bed_markers_test_0_0_0_30_0_0_2500_doubled_mover_tilted_10_y.png")};
        cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
        expect((not image.empty()) >> fatal);

        auto const &cameraMatrix = openScadCameraMatrix2x;

        ProvidedMarkerPositions const effectorMarkerPositions{
            -72.4478, -125.483, 136.03, 72.4478,  -125.483, 136.03,
            146.895,  -3.4642,  136.03, 64.446,   139.34,   136.03,
            -68.4476, 132.411,  136.03, -160.895, -27.7129, 136.03};

        ProvidedMarkerPositions const bedMarkerPositions{
            -300.0, -300.0, 0.0, 250.0, -300.0, 0.0, 250.0,  0.0,   0.0,
            230.0,  380.0,  0.0, 0.0,   400.0,  0.0, -300.0, 380.0, 0.0};

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                        cameraMatrix.at<double>(1, 2)};

        MarkerParams const effectorMarkerParams{effectorMarkerPositions,
                                                markerDiameter};
        MarkerParams const bedMarkerParams{bedMarkerPositions, markerDiameter};

        FinderImage const finderImage{image, meanFocalLength, imageCenter};

        auto const effectorMarks{findMarks(finderImage, effectorMarkerParams,
                                           {.m_showIntermediateImages = false,
                                            .m_verbose = false,
                                            .m_fitByDistance = true})};

        auto const bedMarks{findMarks(finderImage, bedMarkerParams,
                                      {.m_showIntermediateImages = false,
                                       .m_verbose = false,
                                       .m_fitByDistance = true},
                                      {0, 0, -1}, false, effectorMarks)};

        SolvePnpPoints const effectorPoints{
            effectorMarks, markerDiameter,   meanFocalLength,
            imageCenter,   MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        SolvePnpPoints const bedPoints{
            bedMarks,    markerDiameter,   meanFocalLength,
            imageCenter, MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        expect((effectorPoints.allIdentified()) >> fatal);
        expect((bedPoints.allIdentified()) >> fatal);

        std::optional<SixDof> const effectorPoseRelativeToCamera{
            solvePnp(cameraMatrix, effectorMarkerPositions, effectorPoints)};

        std::optional<SixDof> const bedPoseRelativeToCamera{
            solvePnp(cameraMatrix, bedMarkerPositions, bedPoints)};

        expect((effectorPoseRelativeToCamera.has_value()) >> fatal);
        expect((bedPoseRelativeToCamera.has_value()) >> fatal);

        SixDof const pose{
            effectorPoseRelativeToBed(effectorPoseRelativeToCamera.value(),
                                      bedPoseRelativeToCamera.value())};

        auto constexpr EPSX{0.06_d};
        auto constexpr EPSY{0.14_d};
        auto constexpr EPSZ{0.14_d};
        auto constexpr EPSROT{0.01_d};
        auto constexpr EXPECTED_ROT_Y{10.0 * CV_PI/180.0};
        expect(abs(pose.x()) < EPSX) << "translation X";
        expect(abs(pose.y()) < EPSY) << "translation Y";
        expect(abs(pose.z()) < EPSZ) << "translation Z";
        expect(abs(pose.rotX()) < EPSROT) << "rotation X";
        expect(abs(pose.rotY() - EXPECTED_ROT_Y) < EPSROT) << "rotation Y";
        expect(abs(pose.rotZ()) < EPSROT) << "rotation Z";
      };

  "Tilted and twisted effector pose relative to bed pose from OpenScad generated image white disks"_test =
      [&openScadCameraMatrix2x] {
        double constexpr markerDiameter{90.0};
        std::string const imageFileName{hpm::getPath(
            "test-images/"
            "bed_markers_test_0_0_0_30_0_0_2500_doubled_mover_tilted_20_towards_camera_and_twisted_10_deg.png")};
        cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
        expect((not image.empty()) >> fatal);

        auto const &cameraMatrix = openScadCameraMatrix2x;

        ProvidedMarkerPositions const effectorMarkerPositions{
            -72.4478, -125.483, 136.03, 72.4478,  -125.483, 136.03,
            146.895,  -3.4642,  136.03, 64.446,   139.34,   136.03,
            -68.4476, 132.411,  136.03, -160.895, -27.7129, 136.03};

        ProvidedMarkerPositions const bedMarkerPositions{
            -300.0, -300.0, 0.0, 250.0, -300.0, 0.0, 250.0,  0.0,   0.0,
            230.0,  380.0,  0.0, 0.0,   400.0,  0.0, -300.0, 380.0, 0.0};

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                        cameraMatrix.at<double>(1, 2)};

        MarkerParams const effectorMarkerParams{effectorMarkerPositions,
                                                markerDiameter};
        MarkerParams const bedMarkerParams{bedMarkerPositions, markerDiameter};

        FinderImage const finderImage{image, meanFocalLength, imageCenter};

        auto const effectorMarks{findMarks(finderImage, effectorMarkerParams,
                                           {.m_showIntermediateImages = false,
                                            .m_verbose = false,
                                            .m_fitByDistance = true})};

        auto const bedMarks{findMarks(finderImage, bedMarkerParams,
                                      {.m_showIntermediateImages = false,
                                       .m_verbose = false,
                                       .m_fitByDistance = true},
                                      {0, 0, -1}, false, effectorMarks)};

        SolvePnpPoints const effectorPoints{
            effectorMarks, markerDiameter,   meanFocalLength,
            imageCenter,   MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        SolvePnpPoints const bedPoints{
            bedMarks,    markerDiameter,   meanFocalLength,
            imageCenter, MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        expect((effectorPoints.allIdentified()) >> fatal);
        expect((bedPoints.allIdentified()) >> fatal);

        std::optional<SixDof> const effectorPoseRelativeToCamera{
            solvePnp(cameraMatrix, effectorMarkerPositions, effectorPoints)};

        std::optional<SixDof> const bedPoseRelativeToCamera{
            solvePnp(cameraMatrix, bedMarkerPositions, bedPoints)};

        expect((effectorPoseRelativeToCamera.has_value()) >> fatal);
        expect((bedPoseRelativeToCamera.has_value()) >> fatal);

        SixDof const pose{
            effectorPoseRelativeToBed(effectorPoseRelativeToCamera.value(),
                                      bedPoseRelativeToCamera.value())};

        auto constexpr EPSX{0.04_d};
        auto constexpr EPSY{0.06_d};
        auto constexpr EPSZ{0.14_d};
        auto constexpr EPSROT{0.001_d};
        auto constexpr EXPECTED_ROT_X{0.348173}; // Found with rotz(10)*rotx(20) in Matlab
        auto constexpr EXPECTED_ROT_Y{0.0304608};
        auto constexpr EXPECTED_ROT_Z{0.172758};
        expect(abs(pose.x()) < EPSX) << "translation X";
        expect(abs(pose.y()) < EPSY) << "translation Y";
        expect(abs(pose.z()) < EPSZ) << "translation Z";
        expect(abs(pose.rotX() - EXPECTED_ROT_X) < EPSROT) << "rotation X";
        expect(abs(pose.rotY() - EXPECTED_ROT_Y) < EPSROT) << "rotation Y";
        expect(abs(pose.rotZ() - EXPECTED_ROT_Z) < EPSROT) << "rotation Z";
      };

  "Tilted and moved effector pose relative to bed pose from OpenScad generated image white disks"_test =
      [&openScadCameraMatrix2x] {
        double constexpr markerDiameter{90.0};
        std::string const imageFileName{hpm::getPath(
            "test-images/"
            "bed_markers_test_0_0_0_30_0_0_2500_doubled_mover_tilted_and_moved_2.png")};
        cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
        expect((not image.empty()) >> fatal);

        auto const &cameraMatrix = openScadCameraMatrix2x;

        ProvidedMarkerPositions const effectorMarkerPositions{
            -72.4478, -125.483, 136.03, 72.4478,  -125.483, 136.03,
            146.895,  -3.4642,  136.03, 64.446,   139.34,   136.03,
            -68.4476, 132.411,  136.03, -160.895, -27.7129, 136.03};

        ProvidedMarkerPositions const bedMarkerPositions{
            -300.0, -300.0, 0.0, 250.0, -300.0, 0.0, 250.0,  0.0,   0.0,
            230.0,  380.0,  0.0, 0.0,   400.0,  0.0, -300.0, 380.0, 0.0};

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                        cameraMatrix.at<double>(1, 2)};

        MarkerParams const effectorMarkerParams{effectorMarkerPositions,
                                                markerDiameter};
        MarkerParams const bedMarkerParams{bedMarkerPositions, markerDiameter};

        FinderImage const finderImage{image, meanFocalLength, imageCenter};

        auto const effectorMarks{findMarks(finderImage, effectorMarkerParams,
                                           {.m_showIntermediateImages = false,
                                            .m_verbose = false,
                                            .m_fitByDistance = true})};

        auto const bedMarks{findMarks(finderImage, bedMarkerParams,
                                      {.m_showIntermediateImages = false,
                                       .m_verbose = false,
                                       .m_fitByDistance = true},
                                      {0, 0, -1}, false, effectorMarks)};

        SolvePnpPoints const effectorPoints{
            effectorMarks, markerDiameter,   meanFocalLength,
            imageCenter,   MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        SolvePnpPoints const bedPoints{
            bedMarks,    markerDiameter,   meanFocalLength,
            imageCenter, MarkerType::DISK, {0, -0.5, -sqrt(3) / 2}};

        expect((effectorPoints.allIdentified()) >> fatal);
        expect((bedPoints.allIdentified()) >> fatal);

        std::optional<SixDof> const effectorPoseRelativeToCamera{
            solvePnp(cameraMatrix, effectorMarkerPositions, effectorPoints)};

        std::optional<SixDof> const bedPoseRelativeToCamera{
            solvePnp(cameraMatrix, bedMarkerPositions, bedPoints)};

        expect((effectorPoseRelativeToCamera.has_value()) >> fatal);
        expect((bedPoseRelativeToCamera.has_value()) >> fatal);

        SixDof const pose{
            effectorPoseRelativeToBed(effectorPoseRelativeToCamera.value(),
                                      bedPoseRelativeToCamera.value())};

        auto constexpr EPSX{0.04_d};
        auto constexpr EPSY{0.12_d};
        auto constexpr EPSZ{0.18_d};
         auto constexpr EPSROT{0.001_d};
         auto constexpr EXPECTED_ROT_X{20.0 * CV_PI/180.0};
        double constexpr EXPECTED_POS_X{-70.0_d};
        double constexpr EXPECTED_POS_Y{60.0_d};
        double constexpr EXPECTED_POS_Z{10.0_d};
        expect(abs(pose.x() - EXPECTED_POS_X) < EPSX) << "translation X";
        expect(abs(pose.y() - EXPECTED_POS_Y) < EPSY) << "translation Y";
        expect(abs(pose.z() - EXPECTED_POS_Z) < EPSZ) << "translation Z";
        expect(abs(pose.rotX() - EXPECTED_ROT_X) < EPSROT) << "rotation X";
        expect(abs(pose.rotY()) < EPSROT) << "rotation Y";
        expect(abs(pose.rotZ()) < EPSROT) << "rotation Z";
      };

  return 0;
}
