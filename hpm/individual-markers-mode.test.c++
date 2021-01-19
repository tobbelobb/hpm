#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include <gsl/span_ext>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
ENABLE_WARNINGS

#include <boost/ut.hpp> //import boost.ut;

#include <hpm/individual-markers-mode.h++>
#include <hpm/test-util.h++> // getPath
#include <hpm/util.h++>

using namespace hpm;

auto main() -> int {
  using namespace boost::ut;
  // clang-format off
  cv::Mat const openScadCameraMatrix2x = (cv::Mat_<double>(3, 3) << 2 * 3377.17,        0.00, 2 * 1280.0,
                                                                           0.00, 2 * 3378.36, 2 *  671.5,
                                                                           0.00,        0.00,        1.0);
  // clang-format on

  "individual markers positions OpenScad generated image"_test =
      [&openScadCameraMatrix2x] {
        double constexpr knownMarkerDiameter{32.0};
        std::string const imageFileName{hpm::getPath(
            "test-images/"
            "generated_benchmark_nr2_double_res_32_0_0_0_0_0_0_755.png")};
        cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
        expect((not image.empty()) >> fatal);

        std::vector<CameraFramedPosition> const knownPositions{
            {72.4478, 125.483, 755}, {-72.4478, 125.483, 755},
            {144.896, 0, 755},       {72.4478, -125.483, 755},
            {-144.896, 0, 755},      {-72.4478, -125.483, 755}};

        enum IDX : size_t {
          BOTTOMRIGHT = 0,
          BOTTOMLEFT = 1,
          RIGHTEST = 2,
          TOPRIGHT = 3,
          LEFTEST = 4,
          TOPLEFT = 5,
        };
        constexpr size_t NUM_MARKERS{6};
        std::array<std::string, NUM_MARKERS> idxNames{};
        idxNames[RIGHTEST] = "Rightest";
        idxNames[TOPRIGHT] = "Topright";
        idxNames[LEFTEST] = "Leftest";
        idxNames[TOPLEFT] = "Topleft";
        idxNames[BOTTOMRIGHT] = "Bottomright";
        idxNames[BOTTOMLEFT] = "Bottomleft";

        auto const &cameraMatrix = openScadCameraMatrix2x;

        double const meanFocalLength{std::midpoint(
            cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1))};
        PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                        cameraMatrix.at<double>(1, 2)};

        std::vector<CameraFramedPosition> const positions{
            findIndividualMarkerPositions(findMarks(image), knownMarkerDiameter,
                                          meanFocalLength, imageCenter)};

        // This is what we want to test.
        // Given all of the above, are we able to get back the
        // values that we fed into OpenScad?
        auto constexpr EPS{1.2_d};
        // Check the absolute positions first.
        for (gsl::index i{0}; i < std::ssize(positions); ++i) {
          expect(
              (i < std::ssize(knownPositions) and i < std::ssize(idxNames)) >>
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
      };
  "filter marks by distance"_test = [&] {
    // clang-format off
    cv::Mat const cameraMatrix = (cv::Mat_<double>(3, 3) << 3000.0,    0.0, 1000.0,
                                                               0.0, 3000.0, 1000.0,
                                                               0.0,    0.0,    1.0);
    ProvidedMarkerPositions const providedPositions{-1, -1, 0,
                                                     1, -1, 0,
                                                     1,  0, 0,
                                                     1,  1, 0,
                                                    -1,  1, 0,
                                                    -1,  0, 0};
    // clang-format on
    double constexpr knownMarkerDiameter{0.032};
    auto const focalLength{cameraMatrix.at<double>(0, 0)};
    PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                    cameraMatrix.at<double>(1, 2)};
    auto constexpr PIX_DIST{100.0};
    auto const CENTER{cameraMatrix.at<double>(0, 2)};
    auto const F{cameraMatrix.at<double>(0, 0)};
    auto const Z0{F / PIX_DIST};

    auto const MARKER_SIZE_STRAIGHT{
        sphereToEllipseWidthHeight({1, 0, Z0}, focalLength,
                                   knownMarkerDiameter / 2)
            .first};
    auto const MARKER_SIZE_DIAG{
        sphereToEllipseWidthHeight({1, 1, Z0}, focalLength,
                                   knownMarkerDiameter / 2)
            .first};

    // Random false positive red detection result
    KeyPoint const falsePositiveRed{{{200, 500}, 40}};
    // A bit harder to sort out false positive blue detection
    KeyPoint const falsePositiveBlue{{(CENTER + PIX_DIST) * 1.08, CENTER},
                                     MARKER_SIZE_STRAIGHT * 1.08};

    DetectionResult detectionResult{
        {{{CENTER - PIX_DIST, CENTER - PIX_DIST}, MARKER_SIZE_DIAG},
         {{CENTER + PIX_DIST, CENTER - PIX_DIST}, MARKER_SIZE_DIAG},
         falsePositiveRed},
        {{{CENTER + PIX_DIST, CENTER}, MARKER_SIZE_STRAIGHT},
         {{CENTER + PIX_DIST, CENTER + PIX_DIST}, MARKER_SIZE_DIAG}},
        {{{CENTER - PIX_DIST, CENTER - PIX_DIST}, MARKER_SIZE_DIAG},
         falsePositiveBlue,
         {{CENTER + PIX_DIST, CENTER}, MARKER_SIZE_STRAIGHT}}};

    filterMarksByDistance(detectionResult, providedPositions, focalLength,
                          imageCenter, knownMarkerDiameter);
    expect(detectionResult.red.size() == 2_ul);
    expect(detectionResult.red[0] != falsePositiveRed);
    expect(detectionResult.red[1] != falsePositiveRed);
    expect(detectionResult.blue.size() == 2_ul);
    expect(detectionResult.blue[0] != falsePositiveBlue);
    expect(detectionResult.blue[1] != falsePositiveBlue);
  };
  return 0;
}
