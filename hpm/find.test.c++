#include <hpm/find.h++>
#include <hpm/marks.h++>
#include <hpm/test-util.h++> // getPath
#include <hpm/util.h++>

#include <gsl/span_ext>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
ENABLE_WARNINGS

#include <boost/ut.hpp> //import boost.ut;

#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;

  "individual markers positions OpenScad generated image"_test = [] {
    // clang-format off
    cv::Mat const openScadCameraMatrix2x =
      (cv::Mat_<double>(3, 3) << 2 * 3375.85,        0.00, 2 * 1280.0,
                                        0.00, 2 * 3375.85, 2 *  671.5,
                                        0.00,        0.00,        1.0);
    // clang-format on
    double constexpr knownMarkerDiameter{32.0};
    std::string const imageFileName{hpm::getPath(
        "test-images/"
        "generated_benchmark_nr2_double_res_32_0_0_0_0_0_0_755_white.png")};
    cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
    expect((not image.empty()) >> fatal);

    ProvidedMarkerPositions const providedMarkerPositions{
        -72.4478, -125.483, 0.0, 72.4478,  -125.483, 0.0,
        146.895,  -3.4642,  0.0, 64.446,   139.34,   0.0,
        -68.4476, 132.411,  0.0, -160.895, -27.7129, 0.0};

    std::vector<CameraFramedPosition> const knownPositions{
        {-72.4478, 125.483, 755},  {72.4478, 125.483, 755},
        {146.895, 3.4642, 755},    {64.446, -139.34, 755},
        {-68.4476, -132.411, 755}, {-160.895, 27.7129, 755}};

    enum IDX : size_t {
      BOTTOMLEFT = 0,
      BOTTOMRIGHT = 1,
      RIGHTEST = 2,
      TOPRIGHT = 3,
      TOPLEFT = 4,
      LEFTEST = 5,
    };
    constexpr size_t NUM_MARKERS{6};
    std::array<std::string, NUM_MARKERS> idxNames{};
    idxNames[BOTTOMLEFT] = "Bottomleft";
    idxNames[BOTTOMRIGHT] = "Bottomright";
    idxNames[RIGHTEST] = "Rightest";
    idxNames[TOPRIGHT] = "Topright";
    idxNames[TOPLEFT] = "Topleft";
    idxNames[LEFTEST] = "Leftest";

    auto const &cameraMatrix = openScadCameraMatrix2x;

    double const meanFocalLength{std::midpoint(cameraMatrix.at<double>(0, 0),
                                               cameraMatrix.at<double>(1, 1))};
    PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                    cameraMatrix.at<double>(1, 2)};

    MarkerParams const markerParams{providedMarkerPositions,
                                    knownMarkerDiameter};

    FinderImage const finderImage{image, meanFocalLength, imageCenter};

    Marks const marks{findMarks(finderImage, markerParams,
                                {.m_showIntermediateImages = false,
                                 .m_verbose = false,
                                 .m_fitByDistance = true})};
    SolvePnpPoints const points{marks, knownMarkerDiameter / 2.0,
                                meanFocalLength, imageCenter};

    expect((points.allIdentified()) >> fatal);
    std::vector<CameraFramedPosition> const positions{
        findIndividualMarkerPositions(marks, knownMarkerDiameter,
                                      meanFocalLength, imageCenter,
                                      MarkerType::SPHERE)};

    // This is what we want to test.
    // Given all of the above, are we able to get back the
    // values that we fed into OpenScad?
    auto constexpr EPS{1.7_d};
    // Check the absolute positions first.
    for (gsl::index i{0}; i < std::ssize(positions); ++i) {
      expect((i < std::ssize(knownPositions) and i < std::ssize(idxNames)) >>
             fatal);
      auto const position{gsl::at(positions, i)};
      auto const knownPosition{gsl::at(knownPositions, i)};
      auto const idxName{gsl::at(idxNames, i)};

      expect(cv::norm(position - knownPosition) < EPS);
      expect(abs(position.x - knownPosition.x) < EPS)
          << idxName << "x too"
          << ((position.x - knownPosition.x) > 0.0 ? "large" : "small");
      expect(abs(position.y - knownPosition.y) < EPS)
          << idxName << "y too"
          << ((position.y - knownPosition.y) > 0.0 ? "large" : "small");
      expect(abs(position.z - knownPosition.z) < EPS)
          << idxName << "z too"
          << ((position.z - knownPosition.z) > 0.0 ? "large" : "small");
    }

    // If the top check fails, then we want to know in what way it failed.
    // The rest of this test gives a quick such analysis.

    // Check the signs
    expect(positions[RIGHTEST].x > 0 and positions[TOPRIGHT].x > 0)
        << "Blue markers must be on the right";
    expect(positions[LEFTEST].x < 0 and positions[TOPLEFT].x < 0)
        << "Green markers must be on the left";
    expect(positions[BOTTOMLEFT].x < 0.0 and positions[BOTTOMRIGHT].x > 0.0)
        << "Red markers must be left/right";
    for (auto const &pos : positions) {
      expect(pos.z > 0) << "All marker positions must have positive z";
    }
    expect(positions[TOPRIGHT].y < 0.0 and positions[TOPLEFT].y < 0.0)
        << "Top markers should have negative y";
    expect(positions[BOTTOMRIGHT].y > 0.0 and positions[BOTTOMLEFT].y > 0.0)
        << "Bottom (red) markers must have positive y";
  };

  return 0;
}
