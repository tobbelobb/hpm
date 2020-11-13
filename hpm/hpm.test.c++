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

  "generated benchmark nr1"_test = [] {
    std::string const camParamsFileName{
        hpm::getPath("example-cam-params/openscadHandCodedCamParams.xml")};
    std::string const imageFileName{hpm::getPath(
        "test-images/generated_benchmark_nr1_32_0_0_0_45_0_0_755.png")};
    double constexpr knownMarkerDiameter{32.0};
    cv::FileStorage const camParamsFile(camParamsFileName,
                                        cv::FileStorage::READ);
    expect((camParamsFile.isOpened()) >> fatal);

    cv::Mat const cameraMatrix = [&camParamsFile]() {
      cv::Mat cameraMatrix_;
      camParamsFile["camera_matrix"] >> cameraMatrix_;
      return cameraMatrix_;
    }();
    cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
    expect((not image.empty()) >> fatal);

    std::vector<cv::KeyPoint> const detectedMarkers{
        detectMarkers(image, false)};
    expect((detectedMarkers.size() == 6_i) >> fatal); // Must find all markers

    const double meanFocalLength{std::midpoint(cameraMatrix.at<double>(0, 0),
                                               cameraMatrix.at<double>(1, 1))};
    cv::Point2f const imageCenter{
        static_cast<float>(cameraMatrix.at<double>(0, 2)),
        static_cast<float>(cameraMatrix.at<double>(1, 2))};

    auto positions =
        detectedMarkers |
        std::views::transform([&](cv::KeyPoint const &keyPoint) {
          return toCameraPosition(keyPoint, meanFocalLength, imageCenter,
                                  image.size(), knownMarkerDiameter);
        });
    expect(positions[0].x > 0 and positions[1].x > 0)
        << "Blue markers on the right";
    expect(positions[2].x < 0 and positions[3].x < 0)
        << "Green markers on the left";
    expect(std::signbit(positions[4].x * positions[5].x) == 1)
        << "Red markers left/right";

    for (auto const &pos : positions) {
      expect(pos.z > 0);
    }
    auto constexpr EPS{0.01_d};

    expect(std::abs(positions[0].y - positions[2].y) < EPS or
           std::abs(positions[0].y - positions[3].y) < EPS)
        << "Each blue marker has similar y pos as one green marker";
    expect(std::abs(positions[1].y - positions[2].y) < EPS or
           std::abs(positions[1].y - positions[3].y) < EPS)
        << "Each blue marker has similar y pos as one green marker";

    auto constexpr biggerEPS{0.2_d};
    expect(std::abs(cv::norm(positions[0] - positions[2]) -
                    cv::norm(positions[1] - positions[5])) < biggerEPS and
           std::abs(cv::norm(positions[0] - positions[2]) -
                    cv::norm(positions[3] - positions[4])) < biggerEPS)
        << "The three largest crossovers should have the same length";

    double constexpr circle_d{2 * 144.896};
    auto constexpr evenBiggerEPS{0.1_d};
    expect(std::abs(cv::norm(positions[0] - positions[2]) - circle_d) <
               evenBiggerEPS or
           std::abs(cv::norm(positions[0] - positions[3]) - circle_d) <
               evenBiggerEPS or
           std::abs(cv::norm(positions[0] - positions[4]) - circle_d) <
               evenBiggerEPS or
           std::abs(cv::norm(positions[0] - positions[5]) - circle_d) <
               evenBiggerEPS)
        << "Hexagon width should match CAD file";
    expect(std::abs(cv::norm(positions[1] - positions[2]) - circle_d) <
               evenBiggerEPS or
           std::abs(cv::norm(positions[1] - positions[3]) - circle_d) <
               evenBiggerEPS or
           std::abs(cv::norm(positions[1] - positions[4]) - circle_d) <
               evenBiggerEPS or
           std::abs(cv::norm(positions[1] - positions[5]) - circle_d) <
               evenBiggerEPS)
        << "Hexagon width should match CAD file";
    expect(std::abs(cv::norm(positions[2] - positions[0]) - circle_d) <
               evenBiggerEPS or
           std::abs(cv::norm(positions[2] - positions[1]) - circle_d) <
               evenBiggerEPS or
           std::abs(cv::norm(positions[2] - positions[4]) - circle_d) <
               evenBiggerEPS or
           std::abs(cv::norm(positions[2] - positions[5]) - circle_d) <
               evenBiggerEPS)
        << "Hexagon width should match CAD file";
    expect(std::abs(cv::norm(positions[3] - positions[0]) - circle_d) <
               evenBiggerEPS or
           std::abs(cv::norm(positions[3] - positions[1]) - circle_d) <
               evenBiggerEPS or
           std::abs(cv::norm(positions[3] - positions[4]) - circle_d) <
               evenBiggerEPS or
           std::abs(cv::norm(positions[3] - positions[5]) - circle_d) <
               evenBiggerEPS)
        << "Hexagon width should match CAD file";

    // Would really want to just test the absolute positions, but they actually
    // seem to be dead wrong.
    //
    // std::array<Position, 6> const correctWorldPositions{
    //    {144.896, 0, 22},  {72.4478, 125.483, 22},   {-72.4478, 125.483, 22},
    //    {-144.896, 0, 22}, {-72.4478, -125.483, 22}, {72.4478, -125.483, 22}};

    // cv::Mat const cameraRotation {}
    // Position const cameraPosition{0, -755.0 / sqrt(2), 755 / sqrt(2)};

    // for (auto const &pos : positions) {
    //  std::cout << pos << " error: " <<
    //}
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
// Span an inner hexagon of a circle with radius:
// circle_r = 144.896
//
// Camera position
// translate [0,0,0]
// rotate [45,0,0]
// distance 755
// So... at
// [            0,
//   -755/sqrt(2),
//    755/sqrt(2)]
//
// Found positions relative to the camera/camera plane:
// [ 142.799,  15.4505, 728.94 ]
// [ 71.4615,  103.155, 817.945]
// [-143.007,  15.4491, 728.896]
// [-71.7038,  103.155, 817.945]
// [ 71.5839, -72.3039, 643.429]
// [-71.7745, -72.3039, 643.429]
//
// We see that
//
// cam + rel =
//
