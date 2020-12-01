#include <iostream>
#include <numeric>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#pragma GCC diagnostic pop

#include <boost/ut.hpp> //import boost.ut;

#include <hpm/hpm.h++>
#include <hpm/test-util.h++>

auto main(int argc, char **argv) -> int {
  using namespace boost::ut;
  // clang-format off
  cv::Mat const openScadCamParams2x = (cv::Mat_<double>(3, 3) << 2 * 3377.17,        0.00, 2 * 1280.0,
                                                                        0.00, 2 * 3378.36, 2 *  671.5,
                                                                        0.00,        0.00,        1.0);
  // clang-format on
  double constexpr knownMarkerDiameter{32.0};

  "interpretation of results on the OpenScad generated image nr1"_test =
      [&openScadCamParams2x, argc, &argv] {
        // Get the image
        std::string const imageFileName{hpm::getPath(
            "test-images/"
            "generated_benchmark_nr2_double_res_32_0_0_0_0_0_0_755.png")};
        cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
        expect((not image.empty()) >> fatal);

        // This is what we fed into openScad when generating the image
        std::vector<CameraFramedPosition> const knownPositions{
            {72.4478, 125.483, 755},    // red right
            {-72.4478, 125.483, 755},   // red left
            {144.896, 0, 755},          // blue on x-axis
            {72.4478, -125.483, 755},   // blue back
            {-144.896, 0, 755},         // green on x-axis
            {-72.4478, -125.483, 755}}; // green back

        // Let's give the indices names for convenience
        enum IDX : size_t {
          BOTTOMRIGHT = 0,
          BOTTOMLEFT = 1,
          RIGHTEST = 2,
          TOPRIGHT = 3,
          LEFTEST = 4,
          TOPLEFT = 5,
        };
        std::array<std::string, 6> idxNames{};
        idxNames[RIGHTEST] = "Rightest";
        idxNames[TOPRIGHT] = "Topright";
        idxNames[LEFTEST] = "Leftest";
        idxNames[TOPLEFT] = "Topleft";
        idxNames[BOTTOMRIGHT] = "Bottomright";
        idxNames[BOTTOMLEFT] = "Bottomleft";

        // This is what we fed into SimpleBlobDetector together with the
        // generated image
        auto const &cameraMatrix = openScadCamParams2x;

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        cv::Point2f const imageCenter{
            static_cast<float>(cameraMatrix.at<double>(0, 2)),
            static_cast<float>(cameraMatrix.at<double>(1, 2))};

        std::vector<CameraFramedPosition> const positions{
            findIndividualMarkerPositions(image, knownMarkerDiameter,
                                          meanFocalLength, imageCenter, false,
                                          false)};

        // This is what we want to test.
        // Given all of the above, are we able to get back the
        // values that we fed into OpenScad?
        auto constexpr EPS{1.0_d};
        // Check the absolute positions first.
        for (size_t i{0}; i < positions.size(); ++i) {
          assert(i < knownPositions.size() and i < idxNames.size());

          expect(cv::norm(positions[i] - knownPositions[i]) < EPS);
          expect(abs(positions[i].x - knownPositions[i].x) < EPS)
              << idxNames[i] << "x too"
              << ((positions[i].x - knownPositions[i].x) > 0.0 ? "large"
                                                               : "small");
          expect(abs(positions[i].y - knownPositions[i].y) < EPS)
              << idxNames[i] << "y too"
              << ((positions[i].y - knownPositions[i].y) > 0.0 ? "large"
                                                               : "small");
          expect(abs(positions[i].z - knownPositions[i].z) < EPS)
              << idxNames[i] << "z too"
              << ((positions[i].z - knownPositions[i].z) > 0.0 ? "large"
                                                               : "small");
        }

        // If the top check fails, then we want to know in what way it failed.
        // The rest of this test gives a quick such analysis.

        // Check the signs
        expect(positions[RIGHTEST].x > 0 and positions[TOPRIGHT].x > 0)
            << "Blue markers must be on the right";
        expect(positions[LEFTEST].x < 0 and positions[TOPLEFT].x < 0)
            << "Green markers must be on the left";
        expect((positions[BOTTOMLEFT].x < 0.0 and positions[BOTTOMRIGHT].x) >
               0.0)
            << "Red markers must be left/right";
        for (auto const &pos : positions) {
          expect(pos.z > 0) << "All marker positions must have positive z";
        }
        expect(positions[TOPRIGHT].y < 0.0 and positions[TOPLEFT].y < 0.0)
            << "Top markers should have negative y";
        expect(std::abs(positions[RIGHTEST].y) < EPS and
               std::abs(positions[LEFTEST].y) < EPS)
            << "Rightest/leftest markers must be near the xz plane";
        expect(positions[BOTTOMRIGHT].y > 0.0 and positions[BOTTOMLEFT].y > 0.0)
            << "Bottom (red) markers must have positive y";

        // Check that we have some symmetrical properties
        expect(std::abs(positions[RIGHTEST].x + positions[LEFTEST].x) < EPS)
            << "Left/right positions should be mirrored over yz plane";
        expect(std::abs(positions[RIGHTEST].y - positions[LEFTEST].y) < EPS)
            << "Left/right positions should be mirrored over yz plane";
        expect(std::abs(positions[RIGHTEST].z - positions[LEFTEST].z) < EPS)
            << "Left/right positions should be mirrored over yz plane";
        expect(std::abs(positions[TOPRIGHT].x + positions[TOPLEFT].x) < EPS)
            << "Left/right positions should be mirrored over yz plane";
        expect(std::abs(positions[TOPRIGHT].y - positions[TOPLEFT].y) < EPS)
            << "Left/right positions should be mirrored over yz plane";
        expect(std::abs(positions[TOPRIGHT].z - positions[TOPLEFT].z) < EPS)
            << "Left/right positions should be mirrored over yz plane";
        expect(std::abs(positions[BOTTOMRIGHT].x + positions[BOTTOMLEFT].x) <
               EPS)
            << "Left/right positions should be mirrored over yz plane";
        expect(std::abs(positions[BOTTOMRIGHT].y - positions[BOTTOMLEFT].y) <
               EPS)
            << "Left/right positions should be mirrored over yz plane";
        expect(std::abs(positions[BOTTOMRIGHT].z - positions[BOTTOMLEFT].z) <
               EPS)
            << "Left/right positions should be mirrored over yz plane";

        // Check that LEFT/RIGHT direction gets equal treatment as
        // TOPLEFT/BOTTOMRIGHT direction
        expect(std::abs(cv::norm(positions[RIGHTEST] - positions[LEFTEST]) -
                        cv::norm(positions[TOPRIGHT] - positions[BOTTOMLEFT])) <
               EPS)
            << "Largest crossovers should be of equal length";
        expect(std::abs(cv::norm(positions[TOPLEFT] - positions[BOTTOMRIGHT]) -
                        cv::norm(positions[TOPRIGHT] - positions[BOTTOMLEFT])) <
               EPS)
            << "Largest crossovers should be of equal length";

        expect(std::abs(cv::norm(positions[RIGHTEST] - positions[LEFTEST]) -
                        cv::norm(knownPositions[RIGHTEST] -
                                 knownPositions[LEFTEST])) < EPS)
            << "The largest crossovers should have the correct length";

        if (argc > 1 and argv[1][0] == 'f') {
          for (size_t i{0}; i < positions.size(); ++i) {
            assert(i < knownPositions.size());

            std::cout << idxNames[i] << ' ' << positions[i];
            auto const err{positions[i] - knownPositions[i]};
            cv::Point3d const knownPositionXy{knownPositions[i].x,
                                              knownPositions[i].y, 0};
            cv::Point3d const errXy{err.x, err.y, 0};
            if (cv::norm(knownPositionXy) > 0.001 and cv::norm(errXy) > 0.001) {
              cv::Point3d const knownPositionDirection{
                  knownPositions[i] / cv::norm(knownPositions[i])};
              cv::Point3d const knownPositionXyDirection{
                  knownPositionXy / cv::norm(knownPositionXy)};
              cv::Point3d const errDirection{err / cv::norm(err)};
              cv::Point3d const errXyDirection{errXy / cv::norm(errXy)};

              std::cout << " is xy-err towards origin: "
                        << -knownPositionXyDirection.dot(errXyDirection);
              std::cout << " is xyz-err towards origin: "
                        << -knownPositionDirection.dot(errDirection);
            }
            std::cout << '\n';
          }
        }
      };
}
