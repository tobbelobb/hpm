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
  cv::Mat const openScadCamParams6x = (cv::Mat_<double>(3, 3) << 6 * 3375.85,        0.00, 6 * 1280.0,
                                                                        0.00, 6 * 3375.85, 6 *  671.5,
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
                          {.pt = {5119.5, 2685.5}, .size = 572.927}}};

        // This is what we want to test.
        // Given all of the above, are we able to get back the
        // values that we fed into OpenScad?
        auto positions =
            mockedResult.keyPoints |
            std::views::transform([&](cv::KeyPoint const &keyPoint) {
              return blobToCameraPosition(keyPoint, meanFocalLength,
                                          imageCenter, imageSize,
                                          knownMarkerDiameter);
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
              return blobToCameraPosition(keyPoint, meanFocalLength,
                                          imageCenter, imageSize,
                                          knownMarkerDiameter);
            });
        expect(positions.size() == detectedMarkers.size());

        auto constexpr EPS{1.0_d};
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
  "interpretation simpleBlobDetector's results on grid-red-2000.png"_test =
      [&openScadCamParams6x, argc, argv] {
        cv::Size const imageSize{.width = 15360, .height = 8058};
        auto const cameraMatrix = openScadCamParams6x;
        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        cv::Point2f const imageCenter{
            static_cast<float>(cameraMatrix.at<double>(0, 2)),
            static_cast<float>(cameraMatrix.at<double>(1, 2))};

        std::vector<cv::KeyPoint> const detectedMarkers{
            {{14769.3, 7066.97}, 335.594}, {{13756.4, 7066.96}, 333.101},
            {{12743.6, 7066.98}, 330.987}, {{11730.8, 7066.99}, 329.263},
            {{10718, 7066.96}, 328.151},   {{9705.1, 7066.96}, 326.976},
            {{8692.33, 7066.97}, 326.154}, {{7679.51, 7066.98}, 326.054},
            {{6666.69, 7066.95}, 326.215}, {{5653.84, 7066.97}, 326.878},
            {{4641.02, 7066.98}, 328.159}, {{3628.18, 7066.96}, 329.301},
            {{2615.4, 7066.98}, 330.733},  {{1602.57, 7066.97}, 332.916},
            {{589.769, 7066.98}, 335.57},  {{14769.3, 6054.14}, 334.173},
            {{13756.4, 6054.13}, 331.827}, {{12743.6, 6054.12}, 329.795},
            {{11730.8, 6054.13}, 328.249}, {{10718, 6054.11}, 327.024},
            {{9705.12, 6054.12}, 326.039}, {{8692.35, 6054.15}, 325.224},
            {{7679.5, 6054.15}, 324.948},  {{6666.66, 6054.16}, 325.227},
            {{5653.86, 6054.12}, 325.895}, {{4641.02, 6054.13}, 327.016},
            {{3628.21, 6054.13}, 328.311}, {{2615.39, 6054.13}, 329.947},
            {{1602.58, 6054.14}, 332.077}, {{589.748, 6054.13}, 334.238},
            {{14769.3, 5041.33}, 333.348}, {{13756.4, 5041.32}, 331.103},
            {{12743.6, 5041.33}, 329.108}, {{11730.8, 5041.29}, 327.573},
            {{10718, 5041.31}, 326.161},   {{9705.15, 5041.34}, 325.233},
            {{8692.32, 5041.32}, 324.64},  {{7679.5, 5041.31}, 324.428},
            {{6666.69, 5041.32}, 324.612}, {{5653.82, 5041.34}, 325.181},
            {{4641.06, 5041.31}, 326.15},  {{3628.19, 5041.34}, 327.62},
            {{2615.38, 5041.33}, 329.217}, {{1602.56, 5041.31}, 331.397},
            {{589.757, 5041.32}, 333.604}, {{14769.3, 4028.5}, 333.433},
            {{13756.4, 4028.5}, 331.048},  {{12743.6, 4028.5}, 328.428},
            {{11730.8, 4028.5}, 327.342},  {{10718, 4028.5}, 326.009},
            {{9705.17, 4028.5}, 324.975},  {{8692.29, 4028.5}, 324.428},
            {{7679.5, 4028.5}, 324.306},   {{6666.67, 4028.5}, 324.424},
            {{5653.82, 4028.5}, 324.978},  {{4641.05, 4028.5}, 326.063},
            {{3628.2, 4028.5}, 327.157},   {{2615.36, 4028.5}, 328.975},
            {{1602.56, 4028.5}, 331.031},  {{589.742, 4028.5}, 333.435},
            {{14769.3, 3015.67}, 333.348}, {{13756.4, 3015.68}, 331.103},
            {{12743.6, 3015.67}, 329.108}, {{11730.8, 3015.71}, 327.573},
            {{10718, 3015.69}, 326.161},   {{9705.15, 3015.66}, 325.233},
            {{8692.32, 3015.68}, 324.64},  {{7679.5, 3015.69}, 324.428},
            {{6666.69, 3015.68}, 324.612}, {{5653.82, 3015.66}, 325.181},
            {{4641.06, 3015.69}, 326.15},  {{3628.19, 3015.66}, 327.62},
            {{2615.38, 3015.67}, 329.217}, {{1602.56, 3015.69}, 331.397},
            {{589.757, 3015.68}, 333.604}, {{14769.3, 2002.86}, 334.173},
            {{13756.4, 2002.87}, 331.827}, {{12743.6, 2002.88}, 329.793},
            {{11730.8, 2002.87}, 328.249}, {{10718, 2002.89}, 327.024},
            {{9705.12, 2002.88}, 326.039}, {{8692.35, 2002.85}, 325.224},
            {{7679.5, 2002.85}, 324.948},  {{6666.66, 2002.84}, 325.227},
            {{5653.86, 2002.88}, 325.895}, {{4641.02, 2002.87}, 327.016},
            {{3628.21, 2002.87}, 328.311}, {{2615.39, 2002.87}, 329.947},
            {{1602.58, 2002.86}, 332.077}, {{589.748, 2002.87}, 334.238},
            {{14769.3, 990.028}, 335.594}, {{13756.4, 990.037}, 333.1},
            {{12743.6, 990.028}, 330.991}, {{11730.8, 990.012}, 329.264},
            {{10718, 990.04}, 328.151},    {{9705.1, 990.041}, 326.976},
            {{8692.33, 990.029}, 326.159}, {{7679.51, 990.026}, 326.053},
            {{6666.69, 990.046}, 326.215}, {{5653.84, 990.034}, 326.878},
            {{4641.02, 990.023}, 328.159}, {{3628.18, 990.037}, 329.301},
            {{2615.39, 990.025}, 330.735}, {{1602.57, 990.037}, 332.911},
            {{589.769, 990.02}, 335.57}};

        std::vector<Position> const knownPositions{
            {700, 300, 2000},   {600, 300, 2000},   {500, 300, 2000},
            {400, 300, 2000},   {300, 300, 2000},   {200, 300, 2000},
            {100, 300, 2000},   {0, 300, 2000},     {-100, 300, 2000},
            {-200, 300, 2000},  {-300, 300, 2000},  {-400, 300, 2000},
            {-500, 300, 2000},  {-600, 300, 2000},  {-700, 300, 2000},
            {700, 200, 2000},   {600, 200, 2000},   {500, 200, 2000},
            {400, 200, 2000},   {300, 200, 2000},   {200, 200, 2000},
            {100, 200, 2000},   {0, 200, 2000},     {-100, 200, 2000},
            {-200, 200, 2000},  {-300, 200, 2000},  {-400, 200, 2000},
            {-500, 200, 2000},  {-600, 200, 2000},  {-700, 200, 2000},
            {700, 100, 2000},   {600, 100, 2000},   {500, 100, 2000},
            {400, 100, 2000},   {300, 100, 2000},   {200, 100, 2000},
            {100, 100, 2000},   {0, 100, 2000},     {-100, 100, 2000},
            {-200, 100, 2000},  {-300, 100, 2000},  {-400, 100, 2000},
            {-500, 100, 2000},  {-600, 100, 2000},  {-700, 100, 2000},
            {700, 0, 2000},     {600, 0, 2000},     {500, 0, 2000},
            {400, 0, 2000},     {300, 0, 2000},     {200, 0, 2000},
            {100, 0, 2000},     {0, 0, 2000},       {-100, 0, 2000},
            {-200, 0, 2000},    {-300, 0, 2000},    {-400, 0, 2000},
            {-500, 0, 2000},    {-600, 0, 2000},    {-700, 0, 2000},
            {700, -100, 2000},  {600, -100, 2000},  {500, -100, 2000},
            {400, -100, 2000},  {300, -100, 2000},  {200, -100, 2000},
            {100, -100, 2000},  {0, -100, 2000},    {-100, -100, 2000},
            {-200, -100, 2000}, {-300, -100, 2000}, {-400, -100, 2000},
            {-500, -100, 2000}, {-600, -100, 2000}, {-700, -100, 2000},
            {700, -200, 2000},  {600, -200, 2000},  {500, -200, 2000},
            {400, -200, 2000},  {300, -200, 2000},  {200, -200, 2000},
            {100, -200, 2000},  {0, -200, 2000},    {-100, -200, 2000},
            {-200, -200, 2000}, {-300, -200, 2000}, {-400, -200, 2000},
            {-500, -200, 2000}, {-600, -200, 2000}, {-700, -200, 2000},
            {700, -300, 2000},  {600, -300, 2000},  {500, -300, 2000},
            {400, -300, 2000},  {300, -300, 2000},  {200, -300, 2000},
            {100, -300, 2000},  {0, -300, 2000},    {-100, -300, 2000},
            {-200, -300, 2000}, {-300, -300, 2000}, {-400, -300, 2000},
            {-500, -300, 2000}, {-600, -300, 2000}, {-700, -300, 2000}};

        auto positions =
            detectedMarkers |
            std::views::transform([&](cv::KeyPoint const &keyPoint) {
              return blobToCameraPosition(keyPoint, meanFocalLength,
                                          imageCenter, imageSize,
                                          knownMarkerDiameter);
            });

        auto constexpr EPS{3.0_d};
        for (auto i{0}; i < positions.size(); ++i) {
          expect(cv::norm(positions[i] - knownPositions[i]) < EPS)
              << positions[i];
          expect(abs(positions[i].x - knownPositions[i].x) < EPS)
              << positions[i] << knownPositions[i] << " x too"
              << ((positions[i].x - knownPositions[i].x) > 0.0 ? "large"
                                                               : "small");
          expect(abs(positions[i].y - knownPositions[i].y) < EPS)
              << positions[i] << knownPositions[i] << " y too"
              << ((positions[i].y - knownPositions[i].y) > 0.0 ? "large"
                                                               : "small");
          expect(abs(positions[i].z - knownPositions[i].z) < EPS)
              << positions[i] << knownPositions[i] << " z too"
              << ((positions[i].z - knownPositions[i].z) > 0.0 ? "large"
                                                               : "small");
        }
      };

  "interpretation simpleBlobDetector's results on grid-green-2000.png"_test =
      [&openScadCamParams6x, argc, argv] {
        cv::Size const imageSize{.width = 15360, .height = 8058};
        auto const cameraMatrix = openScadCamParams6x;
        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        cv::Point2f const imageCenter{
            static_cast<float>(cameraMatrix.at<double>(0, 2)),
            static_cast<float>(cameraMatrix.at<double>(1, 2))};

        std::vector<cv::KeyPoint> const detectedMarkers{
            {{14769.3, 7066.97}, 335.594}, {{13756.4, 7066.96}, 333.101},
            {{12743.6, 7066.98}, 330.987}, {{11730.8, 7066.99}, 329.263},
            {{10718, 7066.96}, 328.151},   {{9705.1, 7066.96}, 326.976},
            {{8692.33, 7066.97}, 326.154}, {{7679.51, 7066.98}, 326.054},
            {{6666.69, 7066.95}, 326.215}, {{5653.84, 7066.97}, 326.878},
            {{4641.02, 7066.98}, 328.159}, {{3628.18, 7066.96}, 329.301},
            {{2615.4, 7066.98}, 330.733},  {{1602.57, 7066.97}, 332.916},
            {{589.769, 7066.98}, 335.57},  {{14769.3, 6054.14}, 334.173},
            {{13756.4, 6054.13}, 331.827}, {{12743.6, 6054.12}, 329.795},
            {{11730.8, 6054.13}, 328.249}, {{10718, 6054.11}, 327.024},
            {{9705.12, 6054.12}, 326.039}, {{8692.35, 6054.15}, 325.224},
            {{7679.5, 6054.15}, 324.948},  {{6666.66, 6054.16}, 325.227},
            {{5653.86, 6054.12}, 325.895}, {{4641.02, 6054.13}, 327.016},
            {{3628.21, 6054.13}, 328.311}, {{2615.39, 6054.13}, 329.947},
            {{1602.58, 6054.14}, 332.077}, {{589.748, 6054.13}, 334.238},
            {{14769.3, 5041.33}, 333.348}, {{13756.4, 5041.32}, 331.103},
            {{12743.6, 5041.33}, 329.108}, {{11730.8, 5041.29}, 327.573},
            {{10718, 5041.31}, 326.161},   {{9705.15, 5041.34}, 325.233},
            {{8692.32, 5041.32}, 324.64},  {{7679.5, 5041.31}, 324.428},
            {{6666.69, 5041.32}, 324.612}, {{5653.82, 5041.34}, 325.181},
            {{4641.06, 5041.31}, 326.15},  {{3628.19, 5041.34}, 327.62},
            {{2615.38, 5041.33}, 329.217}, {{1602.56, 5041.31}, 331.397},
            {{589.757, 5041.32}, 333.604}, {{14769.3, 4028.5}, 333.433},
            {{13756.4, 4028.5}, 331.048},  {{12743.6, 4028.5}, 328.428},
            {{11730.8, 4028.5}, 327.342},  {{10718, 4028.5}, 326.009},
            {{9705.17, 4028.5}, 324.975},  {{8692.29, 4028.5}, 324.428},
            {{7679.5, 4028.5}, 324.306},   {{6666.67, 4028.5}, 324.424},
            {{5653.82, 4028.5}, 324.978},  {{4641.05, 4028.5}, 326.063},
            {{3628.2, 4028.5}, 327.157},   {{2615.36, 4028.5}, 328.975},
            {{1602.56, 4028.5}, 331.031},  {{589.742, 4028.5}, 333.435},
            {{14769.3, 3015.67}, 333.348}, {{13756.4, 3015.68}, 331.103},
            {{12743.6, 3015.67}, 329.108}, {{11730.8, 3015.71}, 327.573},
            {{10718, 3015.69}, 326.161},   {{9705.15, 3015.66}, 325.233},
            {{8692.32, 3015.68}, 324.64},  {{7679.5, 3015.69}, 324.428},
            {{6666.69, 3015.68}, 324.612}, {{5653.82, 3015.66}, 325.181},
            {{4641.06, 3015.69}, 326.15},  {{3628.19, 3015.66}, 327.62},
            {{2615.38, 3015.67}, 329.217}, {{1602.56, 3015.69}, 331.397},
            {{589.757, 3015.68}, 333.604}, {{14769.3, 2002.86}, 334.173},
            {{13756.4, 2002.87}, 331.827}, {{12743.6, 2002.88}, 329.793},
            {{11730.8, 2002.87}, 328.249}, {{10718, 2002.89}, 327.024},
            {{9705.12, 2002.88}, 326.039}, {{8692.35, 2002.85}, 325.224},
            {{7679.5, 2002.85}, 324.948},  {{6666.66, 2002.84}, 325.227},
            {{5653.86, 2002.88}, 325.895}, {{4641.02, 2002.87}, 327.016},
            {{3628.21, 2002.87}, 328.311}, {{2615.39, 2002.87}, 329.947},
            {{1602.58, 2002.86}, 332.077}, {{589.748, 2002.87}, 334.238},
            {{14769.3, 990.028}, 335.594}, {{13756.4, 990.037}, 333.1},
            {{12743.6, 990.028}, 330.991}, {{11730.8, 990.012}, 329.264},
            {{10718, 990.04}, 328.151},    {{9705.1, 990.041}, 326.976},
            {{8692.33, 990.029}, 326.159}, {{7679.51, 990.026}, 326.053},
            {{6666.69, 990.046}, 326.215}, {{5653.84, 990.034}, 326.878},
            {{4641.02, 990.023}, 328.159}, {{3628.18, 990.037}, 329.301},
            {{2615.39, 990.025}, 330.735}, {{1602.57, 990.037}, 332.911},
            {{589.769, 990.02}, 335.57}};
        std::vector<Position> const knownPositions{
            {700, 300, 2000},   {600, 300, 2000},   {500, 300, 2000},
            {400, 300, 2000},   {300, 300, 2000},   {200, 300, 2000},
            {100, 300, 2000},   {0, 300, 2000},     {-100, 300, 2000},
            {-200, 300, 2000},  {-300, 300, 2000},  {-400, 300, 2000},
            {-500, 300, 2000},  {-600, 300, 2000},  {-700, 300, 2000},
            {700, 200, 2000},   {600, 200, 2000},   {500, 200, 2000},
            {400, 200, 2000},   {300, 200, 2000},   {200, 200, 2000},
            {100, 200, 2000},   {0, 200, 2000},     {-100, 200, 2000},
            {-200, 200, 2000},  {-300, 200, 2000},  {-400, 200, 2000},
            {-500, 200, 2000},  {-600, 200, 2000},  {-700, 200, 2000},
            {700, 100, 2000},   {600, 100, 2000},   {500, 100, 2000},
            {400, 100, 2000},   {300, 100, 2000},   {200, 100, 2000},
            {100, 100, 2000},   {0, 100, 2000},     {-100, 100, 2000},
            {-200, 100, 2000},  {-300, 100, 2000},  {-400, 100, 2000},
            {-500, 100, 2000},  {-600, 100, 2000},  {-700, 100, 2000},
            {700, 0, 2000},     {600, 0, 2000},     {500, 0, 2000},
            {400, 0, 2000},     {300, 0, 2000},     {200, 0, 2000},
            {100, 0, 2000},     {0, 0, 2000},       {-100, 0, 2000},
            {-200, 0, 2000},    {-300, 0, 2000},    {-400, 0, 2000},
            {-500, 0, 2000},    {-600, 0, 2000},    {-700, 0, 2000},
            {700, -100, 2000},  {600, -100, 2000},  {500, -100, 2000},
            {400, -100, 2000},  {300, -100, 2000},  {200, -100, 2000},
            {100, -100, 2000},  {0, -100, 2000},    {-100, -100, 2000},
            {-200, -100, 2000}, {-300, -100, 2000}, {-400, -100, 2000},
            {-500, -100, 2000}, {-600, -100, 2000}, {-700, -100, 2000},
            {700, -200, 2000},  {600, -200, 2000},  {500, -200, 2000},
            {400, -200, 2000},  {300, -200, 2000},  {200, -200, 2000},
            {100, -200, 2000},  {0, -200, 2000},    {-100, -200, 2000},
            {-200, -200, 2000}, {-300, -200, 2000}, {-400, -200, 2000},
            {-500, -200, 2000}, {-600, -200, 2000}, {-700, -200, 2000},
            {700, -300, 2000},  {600, -300, 2000},  {500, -300, 2000},
            {400, -300, 2000},  {300, -300, 2000},  {200, -300, 2000},
            {100, -300, 2000},  {0, -300, 2000},    {-100, -300, 2000},
            {-200, -300, 2000}, {-300, -300, 2000}, {-400, -300, 2000},
            {-500, -300, 2000}, {-600, -300, 2000}, {-700, -300, 2000}};

        auto positions =
            detectedMarkers |
            std::views::transform([&](cv::KeyPoint const &keyPoint) {
              return blobToCameraPosition(keyPoint, meanFocalLength,
                                          imageCenter, imageSize,
                                          knownMarkerDiameter);
            });

        auto constexpr EPS{3.0_d};
        for (auto i{0}; i < positions.size(); ++i) {
          expect(cv::norm(positions[i] - knownPositions[i]) < EPS)
              << positions[i];
          expect(abs(positions[i].x - knownPositions[i].x) < EPS)
              << positions[i] << knownPositions[i] << " x too"
              << ((positions[i].x - knownPositions[i].x) > 0.0 ? "large"
                                                               : "small");
          expect(abs(positions[i].y - knownPositions[i].y) < EPS)
              << positions[i] << knownPositions[i] << " y too"
              << ((positions[i].y - knownPositions[i].y) > 0.0 ? "large"
                                                               : "small");
          expect(abs(positions[i].z - knownPositions[i].z) < EPS)
              << positions[i] << knownPositions[i] << " z too"
              << ((positions[i].z - knownPositions[i].z) > 0.0 ? "large"
                                                               : "small");
        }
      };

  "interpretation simpleBlobDetector's results on grid-blue-2000.png"_test =
      [&openScadCamParams6x, argc, argv] {
        cv::Size const imageSize{.width = 15360, .height = 8058};
        auto const cameraMatrix = openScadCamParams6x;
        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        cv::Point2f const imageCenter{
            static_cast<float>(cameraMatrix.at<double>(0, 2)),
            static_cast<float>(cameraMatrix.at<double>(1, 2))};

        std::vector<cv::KeyPoint> const detectedMarkers{
            {{14769.3, 7066.97}, 335.594}, {{13756.4, 7066.96}, 333.101},
            {{12743.6, 7066.98}, 330.987}, {{11730.8, 7066.99}, 329.263},
            {{10718, 7066.96}, 328.151},   {{9705.1, 7066.96}, 326.976},
            {{8692.33, 7066.97}, 326.154}, {{7679.51, 7066.98}, 326.054},
            {{6666.69, 7066.95}, 326.215}, {{5653.84, 7066.97}, 326.878},
            {{4641.02, 7066.98}, 328.159}, {{3628.18, 7066.96}, 329.301},
            {{2615.4, 7066.98}, 330.733},  {{1602.57, 7066.97}, 332.916},
            {{589.769, 7066.98}, 335.57},  {{14769.3, 6054.14}, 334.173},
            {{13756.4, 6054.13}, 331.827}, {{12743.6, 6054.12}, 329.795},
            {{11730.8, 6054.13}, 328.249}, {{10718, 6054.11}, 327.024},
            {{9705.12, 6054.12}, 326.039}, {{8692.35, 6054.15}, 325.224},
            {{7679.5, 6054.15}, 324.948},  {{6666.66, 6054.16}, 325.227},
            {{5653.86, 6054.12}, 325.895}, {{4641.02, 6054.13}, 327.016},
            {{3628.21, 6054.13}, 328.311}, {{2615.39, 6054.13}, 329.947},
            {{1602.58, 6054.14}, 332.077}, {{589.748, 6054.13}, 334.238},
            {{14769.3, 5041.33}, 333.348}, {{13756.4, 5041.32}, 331.103},
            {{12743.6, 5041.33}, 329.108}, {{11730.8, 5041.29}, 327.573},
            {{10718, 5041.31}, 326.161},   {{9705.15, 5041.34}, 325.233},
            {{8692.32, 5041.32}, 324.64},  {{7679.5, 5041.31}, 324.428},
            {{6666.69, 5041.32}, 324.612}, {{5653.82, 5041.34}, 325.181},
            {{4641.06, 5041.31}, 326.15},  {{3628.19, 5041.34}, 327.62},
            {{2615.38, 5041.33}, 329.217}, {{1602.56, 5041.31}, 331.397},
            {{589.757, 5041.32}, 333.604}, {{14769.3, 4028.5}, 333.433},
            {{13756.4, 4028.5}, 331.048},  {{12743.6, 4028.5}, 328.428},
            {{11730.8, 4028.5}, 327.342},  {{10718, 4028.5}, 326.009},
            {{9705.17, 4028.5}, 324.975},  {{8692.29, 4028.5}, 324.428},
            {{7679.5, 4028.5}, 324.306},   {{6666.67, 4028.5}, 324.424},
            {{5653.82, 4028.5}, 324.978},  {{4641.05, 4028.5}, 326.063},
            {{3628.2, 4028.5}, 327.157},   {{2615.36, 4028.5}, 328.975},
            {{1602.56, 4028.5}, 331.031},  {{589.742, 4028.5}, 333.435},
            {{14769.3, 3015.67}, 333.348}, {{13756.4, 3015.68}, 331.103},
            {{12743.6, 3015.67}, 329.108}, {{11730.8, 3015.71}, 327.573},
            {{10718, 3015.69}, 326.161},   {{9705.15, 3015.66}, 325.233},
            {{8692.32, 3015.68}, 324.64},  {{7679.5, 3015.69}, 324.428},
            {{6666.69, 3015.68}, 324.612}, {{5653.82, 3015.66}, 325.181},
            {{4641.06, 3015.69}, 326.15},  {{3628.19, 3015.66}, 327.62},
            {{2615.38, 3015.67}, 329.217}, {{1602.56, 3015.69}, 331.397},
            {{589.757, 3015.68}, 333.604}, {{14769.3, 2002.86}, 334.173},
            {{13756.4, 2002.87}, 331.827}, {{12743.6, 2002.88}, 329.793},
            {{11730.8, 2002.87}, 328.249}, {{10718, 2002.89}, 327.024},
            {{9705.12, 2002.88}, 326.039}, {{8692.35, 2002.85}, 325.224},
            {{7679.5, 2002.85}, 324.948},  {{6666.66, 2002.84}, 325.227},
            {{5653.86, 2002.88}, 325.895}, {{4641.02, 2002.87}, 327.016},
            {{3628.21, 2002.87}, 328.311}, {{2615.39, 2002.87}, 329.947},
            {{1602.58, 2002.86}, 332.077}, {{589.748, 2002.87}, 334.238},
            {{14769.3, 990.028}, 335.594}, {{13756.4, 990.037}, 333.1},
            {{12743.6, 990.028}, 330.991}, {{11730.8, 990.012}, 329.264},
            {{10718, 990.04}, 328.151},    {{9705.1, 990.041}, 326.976},
            {{8692.33, 990.029}, 326.159}, {{7679.51, 990.026}, 326.053},
            {{6666.69, 990.046}, 326.215}, {{5653.84, 990.034}, 326.878},
            {{4641.02, 990.023}, 328.159}, {{3628.18, 990.037}, 329.301},
            {{2615.39, 990.025}, 330.735}, {{1602.57, 990.037}, 332.911},
            {{589.769, 990.02}, 335.57}};
        std::vector<Position> const knownPositions{
            {700, 300, 2000},   {600, 300, 2000},   {500, 300, 2000},
            {400, 300, 2000},   {300, 300, 2000},   {200, 300, 2000},
            {100, 300, 2000},   {0, 300, 2000},     {-100, 300, 2000},
            {-200, 300, 2000},  {-300, 300, 2000},  {-400, 300, 2000},
            {-500, 300, 2000},  {-600, 300, 2000},  {-700, 300, 2000},
            {700, 200, 2000},   {600, 200, 2000},   {500, 200, 2000},
            {400, 200, 2000},   {300, 200, 2000},   {200, 200, 2000},
            {100, 200, 2000},   {0, 200, 2000},     {-100, 200, 2000},
            {-200, 200, 2000},  {-300, 200, 2000},  {-400, 200, 2000},
            {-500, 200, 2000},  {-600, 200, 2000},  {-700, 200, 2000},
            {700, 100, 2000},   {600, 100, 2000},   {500, 100, 2000},
            {400, 100, 2000},   {300, 100, 2000},   {200, 100, 2000},
            {100, 100, 2000},   {0, 100, 2000},     {-100, 100, 2000},
            {-200, 100, 2000},  {-300, 100, 2000},  {-400, 100, 2000},
            {-500, 100, 2000},  {-600, 100, 2000},  {-700, 100, 2000},
            {700, 0, 2000},     {600, 0, 2000},     {500, 0, 2000},
            {400, 0, 2000},     {300, 0, 2000},     {200, 0, 2000},
            {100, 0, 2000},     {0, 0, 2000},       {-100, 0, 2000},
            {-200, 0, 2000},    {-300, 0, 2000},    {-400, 0, 2000},
            {-500, 0, 2000},    {-600, 0, 2000},    {-700, 0, 2000},
            {700, -100, 2000},  {600, -100, 2000},  {500, -100, 2000},
            {400, -100, 2000},  {300, -100, 2000},  {200, -100, 2000},
            {100, -100, 2000},  {0, -100, 2000},    {-100, -100, 2000},
            {-200, -100, 2000}, {-300, -100, 2000}, {-400, -100, 2000},
            {-500, -100, 2000}, {-600, -100, 2000}, {-700, -100, 2000},
            {700, -200, 2000},  {600, -200, 2000},  {500, -200, 2000},
            {400, -200, 2000},  {300, -200, 2000},  {200, -200, 2000},
            {100, -200, 2000},  {0, -200, 2000},    {-100, -200, 2000},
            {-200, -200, 2000}, {-300, -200, 2000}, {-400, -200, 2000},
            {-500, -200, 2000}, {-600, -200, 2000}, {-700, -200, 2000},
            {700, -300, 2000},  {600, -300, 2000},  {500, -300, 2000},
            {400, -300, 2000},  {300, -300, 2000},  {200, -300, 2000},
            {100, -300, 2000},  {0, -300, 2000},    {-100, -300, 2000},
            {-200, -300, 2000}, {-300, -300, 2000}, {-400, -300, 2000},
            {-500, -300, 2000}, {-600, -300, 2000}, {-700, -300, 2000}};

        auto positions =
            detectedMarkers |
            std::views::transform([&](cv::KeyPoint const &keyPoint) {
              return blobToCameraPosition(keyPoint, meanFocalLength,
                                          imageCenter, imageSize,
                                          knownMarkerDiameter);
            });

        auto constexpr EPS{3.0_d};
        for (auto i{0}; i < positions.size(); ++i) {
          expect(cv::norm(positions[i] - knownPositions[i]) < EPS)
              << positions[i];
          expect(abs(positions[i].x - knownPositions[i].x) < EPS)
              << positions[i] << knownPositions[i] << " x too"
              << ((positions[i].x - knownPositions[i].x) > 0.0 ? "large"
                                                               : "small");
          expect(abs(positions[i].y - knownPositions[i].y) < EPS)
              << positions[i] << knownPositions[i] << " y too"
              << ((positions[i].y - knownPositions[i].y) > 0.0 ? "large"
                                                               : "small");
          expect(abs(positions[i].z - knownPositions[i].z) < EPS)
              << positions[i] << knownPositions[i] << " z too"
              << ((positions[i].z - knownPositions[i].z) > 0.0 ? "large"
                                                               : "small");
        }
      };
}
