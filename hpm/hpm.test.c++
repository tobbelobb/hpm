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

auto sum(auto... args) { return (args + ...); }

auto main(int argc, char **argv) -> int {
  using namespace boost::ut;
  // clang-format off
  cv::Mat const openScadCamParams4x = (cv::Mat_<double>(3, 3) << 4 * 3377.17,        0.00, 4 * 1280.0,
                                                                        0.00, 4 * 3378.36, 4 *  671.5,
                                                                        0.00,        0.00,        1.0);
  cv::Mat const openScadCamParams6x = (cv::Mat_<double>(3, 3) << 6 * 3375.85,        0.00, 6 * 1280.0,
                                                                        0.00, 6 * 3375.85, 6 *  671.5,
                                                                        0.00,        0.00,        1.0);
  // clang-format on
  double constexpr knownMarkerDiameter{32.0};

  "interpretation of simpleBlobDetector's results on the OpenScad generated image nr1"_test =
      [&openScadCamParams4x, argc, &argv] {
        // This is what we fed into openScad when generating the image
        std::vector<CameraFramedPosition> const knownPositions{
            {144.896, 0, 733},         // blue on x-axis
            {72.4478, -125.483, 733},  // blue back
            {-144.896, 0, 733},        // green on x-axis
            {-72.4478, -125.483, 733}, // green back
            {72.4478, 125.483, 733},   // red right
            {-72.4478, 125.483, 733},  // red left
            {0, 0, 755}};              // center

        // Let's give the indices names for convenience
        enum IDX : size_t {
          RIGHTEST = 0,
          TOPRIGHT = 1,
          LEFTEST = 2,
          TOPLEFT = 3,
          BOTTOMRIGHT = 4,
          BOTTOMLEFT = 5,
          CENTER = 6
        };
        std::array<std::string, 7> idxNames{};
        idxNames[RIGHTEST] = "Rightest";
        idxNames[TOPRIGHT] = "Topright";
        idxNames[LEFTEST] = "Leftest";
        idxNames[TOPLEFT] = "Topleft";
        idxNames[BOTTOMRIGHT] = "Bottomright";
        idxNames[BOTTOMLEFT] = "Bottomleft";
        idxNames[CENTER] = "Center";

        // This is what we fed into SimpleBlobDetector together with the
        // generated image
        auto const &cameraMatrix = openScadCamParams4x;

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        cv::Point2f const imageCenter{
            static_cast<float>(cameraMatrix.at<double>(0, 2)),
            static_cast<float>(cameraMatrix.at<double>(1, 2))};

        // This is what SimpleBlobDetector gave us back
        std::vector<cv::KeyPoint> const mockedResult{
            {{7790.05F, 2685.5F}, 595.324F},  {{6454.76F, 372.741F}, 595.764F},
            {{2448.95F, 2685.5F}, 595.324F},  {{3784.24F, 372.741F}, 595.764F},
            {{6454.77F, 4998.27F}, 595.762F}, {{3784.23F, 4998.27F}, 595.762F},
            {{5119.5F, 2685.5F}, 572.927F}};

        // This is what we want to test.
        // Given all of the above, are we able to get back the
        // values that we fed into OpenScad?
        auto const positions =
            mockedResult | std::views::transform([&](cv::KeyPoint const &blob) {
              return blobToPosition(blob, meanFocalLength, imageCenter,
                                    knownMarkerDiameter);
            });

        auto constexpr EPS{0.448_d};
        // Check the absolute positions first.
        for (auto i{0}; i < std::ssize(positions); ++i) {
          auto const iu{static_cast<size_t>(i)};
          assert(iu < knownPositions.size() and iu < idxNames.size());

          expect(cv::norm(positions[i] - knownPositions[iu]) < EPS);
          expect(abs(positions[i].x - knownPositions[iu].x) < EPS)
              << idxNames[iu] << "x too"
              << ((positions[i].x - knownPositions[iu].x) > 0.0 ? "large"
                                                                : "small");
          expect(abs(positions[i].y - knownPositions[iu].y) < EPS)
              << idxNames[iu] << "y too"
              << ((positions[i].y - knownPositions[iu].y) > 0.0 ? "large"
                                                                : "small");
          expect(abs(positions[i].z - knownPositions[iu].z) < EPS)
              << idxNames[iu] << "z too"
              << ((positions[i].z - knownPositions[iu].z) > 0.0 ? "large"
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
          for (auto i{0}; i < std::ssize(positions); ++i) {
            auto const iu{static_cast<size_t>(i)};
            assert(iu < knownPositions.size());

            std::cout << idxNames[iu] << ' ' << positions[i];
            auto const err{positions[i] - knownPositions[iu]};
            cv::Point3d const knownPositionXy{knownPositions[iu].x,
                                              knownPositions[iu].y, 0};
            cv::Point3d const errXy{err.x, err.y, 0};
            if (cv::norm(knownPositionXy) > 0.001 and cv::norm(errXy) > 0.001) {
              cv::Point3d const knownPositionDirection{
                  knownPositions[iu] / cv::norm(knownPositions[iu])};
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

  "interpretation simpleBlobDetector's results on the OpenScad generated 123Mpx benchmark image"_test =
      [&openScadCamParams6x, argc, argv] {
        auto const cameraMatrix = openScadCamParams6x;
        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        cv::Point2f const imageCenter{
            static_cast<float>(cameraMatrix.at<double>(0, 2)),
            static_cast<float>(cameraMatrix.at<double>(1, 2))};

        std::vector<cv::KeyPoint> const detectedBlobs{
            {.pt = {10615.1F, 4028.5F}, .size = 651.901F},
            {.pt = {9147.31F, 1486.16F}, .size = 652.191F},
            {.pt = {4743.87F, 4028.5F}, .size = 651.847F},
            {.pt = {6211.69F, 1486.17F}, .size = 652.185F},
            {.pt = {9147.33F, 6570.83F}, .size = 652.169F},
            {.pt = {6211.7F, 6570.83F}, .size = 652.223F},
            {.pt = {7679.5F, 4028.5F}, .size = 648.853F}};
        std::vector<CameraFramedPosition> const knownPositions{
            {144.896, 0, 1000},         // blue on x-axis
            {72.4478, -125.483, 1000},  // blue back
            {-144.896, 0, 1000},        // green on x-axis
            {-72.4478, -125.483, 1000}, // green back
            {72.4478, 125.483, 1000},   // red right
            {-72.4478, 125.483, 1000},  // red left
            {0, 0, 1000}};              // center
        enum IDX : size_t {
          RIGHTEST = 0,
          TOPRIGHT = 1,
          LEFTEST = 2,
          TOPLEFT = 3,
          BOTTOMRIGHT = 4,
          BOTTOMLEFT = 5,
          CENTER = 6
        };
        std::array<std::string, 7> idxNames{};
        idxNames[RIGHTEST] = "Rightest";
        idxNames[TOPRIGHT] = "Topright";
        idxNames[LEFTEST] = "Leftest";
        idxNames[TOPLEFT] = "Topleft";
        idxNames[BOTTOMRIGHT] = "Bottomright";
        idxNames[BOTTOMLEFT] = "Bottomleft";
        idxNames[CENTER] = "Center";

        expect(knownPositions.size() == detectedBlobs.size());

        auto const positions =
            detectedBlobs |
            std::views::transform([&](cv::KeyPoint const &keyPoint) {
              return blobToPosition(keyPoint, meanFocalLength, imageCenter,
                                    knownMarkerDiameter);
            });

        auto constexpr EPS{1.0_d};
        for (auto i{0}; i < std::ssize(positions); ++i) {
          auto const iu{static_cast<size_t>(i)};
          assert(iu < knownPositions.size() and iu < idxNames.size());

          expect(cv::norm(positions[i] - knownPositions[iu]) < EPS);
          expect(abs(positions[i].x - knownPositions[iu].x) < EPS)
              << idxNames[iu] << "x too"
              << ((positions[i].x - knownPositions[iu].x) > 0.0 ? "large"
                                                                : "small");
          expect(abs(positions[i].y - knownPositions[iu].y) < EPS)
              << idxNames[iu] << "y too"
              << ((positions[i].y - knownPositions[iu].y) > 0.0 ? "large"
                                                                : "small");
          expect(abs(positions[i].z - knownPositions[iu].z) < EPS)
              << idxNames[iu] << "z too"
              << ((positions[i].z - knownPositions[iu].z) > 0.0 ? "large"
                                                                : "small");
        }

        // A little analysis of the kind of error we get
        if (argc > 1 and argv[1][0] == 's') {
          for (auto i{0}; i < std::ssize(positions); ++i) {
            auto const iu{static_cast<size_t>(i)};
            assert(iu < knownPositions.size());

            auto const err{positions[i] - knownPositions[iu]};
            cv::Point3d const knownPositionXy{knownPositions[iu].x,
                                              knownPositions[iu].y, 0};

            std::cout << std::fixed << std::setprecision(5) << positions[i]
                      << err << std::left << " norm: " << cv::norm(err);
            cv::Point3d const errXy{err.x, err.y, 0};
            if (cv::norm(knownPositionXy) > 0.001 and cv::norm(errXy) > 0.001) {
              cv::Point3d const knownPositionXyDirection{
                  knownPositionXy / cv::norm(knownPositionXy)};
              cv::Point3d const errXyDirection{errXy / cv::norm(errXy)};
              std::cout << " pointing center?: "
                        << -knownPositionXyDirection.dot(errXyDirection)
                        << " xy-err fraction: "
                        << (abs(err.x) + abs(err.y)) /
                               (abs(err.x) + abs(err.y) + abs(err.z));
            }
            std::cout << '\n';
          }
        }
      };
}
