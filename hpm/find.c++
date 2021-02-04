#include <hpm/ellipse-detector.h++>
#include <hpm/find.h++>
#include <hpm/util.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
ENABLE_WARNINGS

#include <algorithm>
#include <cmath> // atan
#include <limits>
#include <ranges>
#include <string>

using namespace hpm;

auto find(cv::InputArray undistortedImage,
          hpm::ProvidedMarkerPositions const &markPos, double const focalLength,
          hpm::PixelPosition const &imageCenter, double const markerDiameter,
          bool showIntermediateImages, bool verbose, bool fitByDistance)
    -> FindResult {
  Marks const marks{findMarks(undistortedImage, markPos, focalLength,
                              imageCenter, markerDiameter,
                              showIntermediateImages, verbose, fitByDistance)};
  SolvePnpPoints const points{marks, markerDiameter / 2.0, focalLength,
                              imageCenter};
  return {points, marks};
}

static auto bgr2skewedHue(cv::Point3_<uint8_t> bgr) -> double {
  auto smallHue = [](std::array<double, 3> const &rgb) -> double {
    auto const [min, max] = std::minmax_element(std::begin(rgb), std::end(rgb));
    auto const diff{std::max(std::numeric_limits<double>::min(), *max - *min)};
    if (max == std::begin(rgb)) {
      return (rgb[1] - rgb[2]) / diff;
    }
    if (max == std::next(std::begin(rgb))) {
      return 2.0 + (rgb[2] - rgb[0]) / diff;
    }
    return 4.0 + (rgb[0] - rgb[1]) / diff;
  };
  std::array<double, 3> const rgb = {static_cast<double>(bgr.z) / 255.0,
                                     static_cast<double>(bgr.y) / 255.0,
                                     static_cast<double>(bgr.x) / 255.0};
  constexpr double MAX_DEG{360.0};
  double hue = smallHue(rgb);
  hue *= 60.0;
  if (hue < 0.0) {
    hue += MAX_DEG;
  }
  if (not(hue < MAX_DEG) or not(hue >= 0.0)) {
    std::cerr << "HUE WAS " << hue << "!!!\n";
  }
  assert(hue < MAX_DEG);
  assert(hue >= 0.0);
  double const ret = std::fmod(hue + 60.0, MAX_DEG);
  // skew hue so all red colors end up closer to 0 than to MAX_DEG.
  return ret;
}

auto findMarks(cv::InputArray undistortedImage,
               hpm::ProvidedMarkerPositions const &markPos,
               double const focalLength, PixelPosition const &imageCenter,
               double const markerDiameter, bool showIntermediateImages,
               bool verbose, bool fitByDistance) -> Marks {

  auto ellipses{ellipseDetect(undistortedImage, showIntermediateImages)};
  if (ellipses.empty()) {
    return {};
  }
  cv::Mat imageMat{undistortedImage.getMat()};

  std::vector<hpm::CameraFramedPosition> positions;
  for (auto const &e : ellipses) {
    positions.emplace_back(
        e.toPosition(focalLength, imageCenter, markerDiameter));
  }
  double bubbleSizeLimit{0.0};
  for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
    for (size_t j{i + 1}; j < NUMBER_OF_MARKERS; ++j) {
      double const dist = cv::norm(markPos.row(static_cast<int>(i)) -
                                   markPos.row(static_cast<int>(j)));
      if (dist > bubbleSizeLimit) {
        bubbleSizeLimit = dist;
      }
    }
  }
  bubbleSizeLimit *= 1.1;
  std::vector<size_t> validEllipseIndices;
  std::vector<size_t> ellipseNeighCounts(positions.size(), 0);
  for (size_t i{0}; i < positions.size(); ++i) {
    for (size_t j{0}; j < positions.size(); ++j) {
      if (i != j and cv::norm(positions[i] - positions[j]) < bubbleSizeLimit) {
        ellipseNeighCounts[i] = ellipseNeighCounts[i] + 1;
        if (ellipseNeighCounts[i] >= 5) {
          validEllipseIndices.emplace_back(i);
          break;
        }
      }
    }
  }

  std::vector<hpm::Mark> marks;
  if (fitByDistance) {
    for (auto const &ellipseIndex : validEllipseIndices) {
      auto const e{ellipses[ellipseIndex]};
      // The ellipse center might be outside of picture frame
      int const row{
          std::clamp(static_cast<int>(e.m_center.y), 0, imageMat.rows - 1)};
      int const col{
          std::clamp(static_cast<int>(e.m_center.x), 0, imageMat.cols - 1)};
      auto bgr = imageMat.at<cv::Point3_<uint8_t>>(row, col);

      if (not((bgr.x > 220 and bgr.y > 220 and bgr.z > 220) or
              (bgr.x < 35 and bgr.y < 35 and bgr.z < 35))) {
        marks.emplace_back(e, bgr2skewedHue(bgr));
      }
    }
    if (showIntermediateImages) {
      cv::Mat cpy = imageMat.clone();
      const auto AQUA{cv::Scalar(255, 255, 0)};
      for (auto const &mark : marks) {
        draw(cpy, mark, AQUA);
      }
      showImage(cpy, "distance-bubble-sorted-ones.png");
    }
  } else {
    for (auto const &ellipse : ellipses) {
      marks.emplace_back(ellipse, 0.0);
    }
  }

  std::sort(marks.begin(), marks.end(), [&](auto const &lhv, auto const &rhv) {
    return lhv.m_hue < rhv.m_hue;
  });

  if (marks.empty()) {
    return {};
  }

  double const min = marks[0].m_hue;
  double const max = marks.back().m_hue;
  double redMid = min;
  double greenMid = marks[marks.size() / 2].m_hue;
  double blueMid = max;

  Marks result{};
  for (auto const &mark : marks) {
    double colorDistanceRed = std::abs(mark.m_hue - redMid);
    double colorDistanceGreen = std::abs(mark.m_hue - greenMid);
    double colorDistanceBlue = std::abs(mark.m_hue - blueMid);
    if (colorDistanceRed < colorDistanceGreen and
        colorDistanceRed < colorDistanceBlue) {
      result.m_red.emplace_back(mark);
    } else if (colorDistanceGreen < colorDistanceRed and
               colorDistanceGreen < colorDistanceBlue) {
      result.m_green.emplace_back(mark);
    } else {
      result.m_blue.emplace_back(mark);
    }
  }

  if (showIntermediateImages) {
    showImage(imageWith(undistortedImage, result),
              "found-marks-before-fit-by-distance.png");
  }
  if (fitByDistance) {
    result.fit(markPos, focalLength, imageCenter, markerDiameter);
    if (showIntermediateImages) {
      showImage(imageWith(undistortedImage, result),
                "found-marks-after-fit-by-distance.png");
    }
  }
  if (verbose) {
    std::cout << "Found " << result.m_red.size() << " red markers, "
              << result.m_green.size() << " green markers, and "
              << result.m_blue.size() << " blue markers\n";
  }

  return result;
}

auto findIndividualMarkerPositions(Marks const &marks,
                                   double const knownMarkerDiameter,
                                   double const focalLength,
                                   PixelPosition const &imageCenter)
    -> std::vector<CameraFramedPosition> {
  std::vector<CameraFramedPosition> positions{};
  positions.reserve(marks.size());
  for (auto const &detected : marks.m_red) {
    positions.emplace_back(
        detected.toPosition(focalLength, imageCenter, knownMarkerDiameter));
  }
  for (auto const &detected : marks.m_green) {
    positions.emplace_back(
        detected.toPosition(focalLength, imageCenter, knownMarkerDiameter));
  }
  for (auto const &detected : marks.m_blue) {
    positions.emplace_back(
        detected.toPosition(focalLength, imageCenter, knownMarkerDiameter));
  }

  return positions;
}
