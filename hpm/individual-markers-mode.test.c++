#include <iostream>
#include <numeric>
#include <vector>

#include <gsl/span_ext>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#pragma GCC diagnostic pop

#include <boost/ut.hpp> //import boost.ut;

#include <hpm/individual-markers-mode.h++>
#include <hpm/test-util.h++> // getPath

using namespace hpm;

auto main() -> int {
  using namespace boost::ut;
  // clang-format off
  cv::Mat const openScadCameraMatrix2x = (cv::Mat_<double>(3, 3) << 2 * 3377.17,        0.00, 2 * 1280.0,
                                                                           0.00, 2 * 3378.36, 2 *  671.5,
                                                                           0.00,        0.00,        1.0);
  // clang-format on
  double constexpr knownMarkerDiameter{32.0};

  "individual markers positions OpenScad generated image"_test =
      [&openScadCameraMatrix2x] {
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
        auto constexpr EPS{1.0_d};
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
  return 0;
}
