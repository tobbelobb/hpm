#include <iostream>
#include <numeric>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <boost/ut.hpp> //import boost.ut;

#include <hpm/hpm.h++>
#include <hpm/test-util.h++>

auto sum(auto... args) { return (args + ...); }

int main(int argc, char **argv) {
  using namespace boost::ut;
  // clang-format off
  // For some reason, the larger focal length measurements
  // gives better results
  cv::Mat const openScadCamParams4x = (cv::Mat_<double>(3, 3) << 4 * 3377.17,        0.00, 4 * 1280.0,
                                                                        0.00, 4 * 3378.36, 4 *  671.5,
                                                                        0.00,        0.00,        1.0);
  cv::Mat const openScadCamParams6x = (cv::Mat_<double>(3, 3) << 6 * 3377.17,        0.00, 6 * 1280.0,
                                                                        0.00, 6 * 3378.36, 6 *  671.5,
                                                                        0.00,        0.00,        1.0);
  // clang-format on
  double constexpr knownMarkerDiameter{32.0};

  "interpretation of simpleBlobDetector's results on the OpenScad generated image nr1"_test =
      [&openScadCamParams4x, argc, &argv] {
        // This is what we fed into openScad when generating the image
        cv::Size const imageSize{.width = 10240, .height = 5372};
        std::vector<Position> const knownPositions{
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
        auto const cameraMatrix = openScadCamParams4x;

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        cv::Point2f const imageCenter{
            static_cast<float>(cameraMatrix.at<double>(0, 2)),
            static_cast<float>(cameraMatrix.at<double>(1, 2))};

        // This is what SimpleBlobDetector gave us back
        detectionResult const mockedResult{
            .keyPoints = {{.pt = {7790.05, 2685.5}, .size = 595.324},
                          {.pt = {6454.76, 372.741}, .size = 595.764},
                          {.pt = {2448.95, 2685.5}, .size = 595.324},
                          {.pt = {3784.24, 372.741}, .size = 595.764},
                          {.pt = {6454.77, 4998.27}, .size = 595.762},
                          {.pt = {3784.23, 4998.27}, .size = 595.762},
                          {.pt = {5119.5, 2685.5}, .size = 572.927}},
            .ellipsenessInclusion = 0.50};

        // This is what we want to test.
        // Given all of the above, are we able to get back the
        // values that we fed into OpenScad?
        auto positions =
            mockedResult.keyPoints |
            std::views::transform([&](cv::KeyPoint const &keyPoint) {
              return toCameraPosition(keyPoint, meanFocalLength, imageCenter,
                                      imageSize, knownMarkerDiameter,
                                      mockedResult.ellipsenessInclusion);
            });

        auto constexpr EPS{0.448_d};
        // Check the absolute positions first.
        for (auto i{0}; i < positions.size(); ++i) {
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
          for (auto i{0}; i < positions.size(); ++i) {
            std::cout << idxNames[i] << ' ' << positions[i];
            auto const err{positions[i] - knownPositions[i]};
            Position const knownPositionXy{knownPositions[i].x,
                                           knownPositions[i].y, 0};
            Position const errXy{err.x, err.y, 0};
            if (cv::norm(knownPositionXy) > 0.001 and cv::norm(errXy) > 0.001) {
              Position const knownPositionDirection{
                  knownPositions[i] / cv::norm(knownPositions[i])};
              Position const knownPositionXyDirection{
                  knownPositionXy / cv::norm(knownPositionXy)};
              Position const errDirection{err / cv::norm(err)};
              Position const errXyDirection{errXy / cv::norm(errXy)};

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
        cv::Size const imageSize{.width = 15360, .height = 8058};
        auto const cameraMatrix = openScadCamParams6x;
        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        cv::Point2f const imageCenter{
            static_cast<float>(cameraMatrix.at<double>(0, 2)),
            static_cast<float>(cameraMatrix.at<double>(1, 2))};

        std::vector<cv::KeyPoint> const detectedMarkers{
            {.pt = {10615.1, 4028.5}, .size = 651.901},
            {.pt = {9147.31, 1486.16}, .size = 652.191},
            {.pt = {4743.87, 4028.5}, .size = 651.847},
            {.pt = {6211.69, 1486.17}, .size = 652.185},
            {.pt = {9147.33, 6570.83}, .size = 652.169},
            {.pt = {6211.7, 6570.83}, .size = 652.223},
            {.pt = {7679.5, 4028.5}, .size = 648.853}};
        double constexpr detectorElipsenessInclusion{0.5};
        std::vector<Position> const knownPositions{
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

        expect(knownPositions.size() == detectedMarkers.size());

        auto positions =
            detectedMarkers |
            std::views::transform([&](cv::KeyPoint const &keyPoint) {
              return toCameraPosition(keyPoint, meanFocalLength, imageCenter,
                                      imageSize, knownMarkerDiameter,
                                      detectorElipsenessInclusion);
            });
        expect(positions.size() == detectedMarkers.size());

        auto constexpr EPS{0.377_d};
        for (auto i{0}; i < positions.size(); ++i) {
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

        // A little analysis of the kind of error we get
        if (argc > 1 and argv[1][0] == 's') {
          for (auto i{0}; i < knownPositions.size(); ++i) {
            auto const err{positions[i] - knownPositions[i]};
            Position const knownPositionXy{knownPositions[i].x,
                                           knownPositions[i].y, 0};

            std::cout << std::fixed << std::setprecision(5) << positions[i]
                      << err << std::left << " norm: " << cv::norm(err);
            Position const errXy{err.x, err.y, 0};
            if (cv::norm(knownPositionXy) > 0.001 and cv::norm(errXy) > 0.001) {
              Position const knownPositionXyDirection{
                  knownPositionXy / cv::norm(knownPositionXy)};
              Position const errXyDirection{errXy / cv::norm(errXy)};
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

// Real positions of markers
// blue0:  [ 144.896 ,    0    , 22]
// blue1:  [  72.4478,  125.483, 22]
// green0: [ -72.4478,  125.483, 22]
// green1: [-144.896 ,    0    , 22]
// red0:   [ -72.4478, -125.483, 22]
// red1:   [  72.4478, -125.483, 22]
//
// Span a circle's inner hexagon
// circle_r = 144.896
