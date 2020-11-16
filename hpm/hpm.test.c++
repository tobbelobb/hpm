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

int main() {
  using namespace boost::ut;
  // clang-format off
  cv::Mat const openScadCamParams   = (cv::Mat_<double>(3, 3) <<     3375.85,        0.00,     1280.0,
                                                                        0.00,     3375.85,      671.5,
                                                                        0.00,        0.00,        1.0);
  cv::Mat const openScadCamParams2x = (cv::Mat_<double>(3, 3) << 2 * 3375.85,        0.00, 2 * 1280.0,
                                                                        0.00, 2 * 3375.85, 2 *  671.5,
                                                                        0.00,        0.00,        1.0);
  cv::Mat const openScadCamParams4x = (cv::Mat_<double>(3, 3) << 4 * 3375.85,        0.00, 4 * 1280.0,
                                                                        0.00, 4 * 3375.85, 4 *  671.5,
                                                                        0.00,        0.00,        1.0);
  cv::Mat const openScadCamParams6x = (cv::Mat_<double>(3, 3) << 6 * 3375.85,        0.00, 6 * 1280.0,
                                                                        0.00, 6 * 3375.85, 6 *  671.5,
                                                                        0.00,        0.00,        1.0);
  // clang-format on
  double constexpr knownMarkerDiameter{32.0};

  "mocked 123Mpx benchmark"_test = [&openScadCamParams6x] {
    cv::Size const imageSize{.width = 15360, .height = 8058};
    auto const cameraMatrix = openScadCamParams6x;
    double const meanFocalLength{std::midpoint(cameraMatrix.at<double>(0, 0),
                                               cameraMatrix.at<double>(1, 1))};
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
    expect(knownPositions.size() == detectedMarkers.size());

    auto positions =
        detectedMarkers |
        std::views::transform([&](cv::KeyPoint const &keyPoint) {
          return toCameraPosition(keyPoint, meanFocalLength, imageCenter,
                                  imageSize, knownMarkerDiameter);
        });
    expect(positions.size() == detectedMarkers.size());

    // A little analysis of the kind of error we get
    for (auto i{0}; i < knownPositions.size(); ++i) {
      auto const err{positions[i] - knownPositions[i]};
      Position const knownPositionXy{knownPositions[i].x, knownPositions[i].y,
                                     0};

      std::cout << std::fixed << std::setprecision(5) << positions[i] << err
                << std::left << " norm: " << cv::norm(err);
      Position const errXy{err.x, err.y, 0};
      if (cv::norm(knownPositionXy) > 0.001 and cv::norm(errXy) > 0.001) {
        Position const knownPositionXyDirection{knownPositionXy /
                                                cv::norm(knownPositionXy)};
        Position const errXyDirection{errXy / cv::norm(errXy)};
        std::cout << " pointing center?: "
                  << -knownPositionXyDirection.dot(errXyDirection)
                  << " xy-err fraction: "
                  << (abs(err.x) + abs(err.y)) /
                         (abs(err.x) + abs(err.y) + abs(err.z));
      }
      std::cout << '\n';
    }
    // Camera: [144.428, -0.0244592, 998.988]mm
    // Camera: [72.1696, -125.07, 998.547]mm
    // Camera: [-144.49, -0.0244132, 999.072]mm
    // Camera: [-72.2194, -125.07, 998.557]mm
    // Camera: [72.173, 125.024, 998.579]mm
    // Camera: [-72.2149, 125.014, 998.498]mm
    // Camera: [-0.0246621, -0.0246621, 999.066]mm
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
