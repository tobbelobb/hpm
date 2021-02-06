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

static auto bgr2hue(cv::Point3_<uint8_t> bgr) -> double {
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
  return hue;
}

static auto getLargest(std::vector<double> const &v, int howManyToSort)
    -> std::vector<size_t> {
  struct Comp {
    Comp(const std::vector<double> &v) : _v(v) {}
    bool operator()(size_t a, size_t b) { return _v[a] > _v[b]; }
    const std::vector<double> &_v;
  };
  std::vector<size_t> largest;
  largest.resize(v.size());
  for (size_t i = 0; i < largest.size(); ++i) {
    largest[i] = i;
  }
  std::partial_sort(largest.begin(), std::next(largest.begin(), howManyToSort),
                    largest.end(), Comp(v));
  largest.resize(static_cast<size_t>(howManyToSort));
  return largest;
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
  bubbleSizeLimit *= 1.7;
  std::vector<size_t> validEllipseIndices;
  std::vector<std::vector<size_t>> ellipseNeighs(positions.size(),
                                                 std::vector<size_t>{});
  double const bubbleSizeLimitSq{bubbleSizeLimit * bubbleSizeLimit};
  for (size_t i{0}; i < positions.size(); ++i) {
    for (size_t j{0}; j < positions.size(); ++j) {
      auto const pixDist{cv::norm(ellipses[i].m_center - ellipses[j].m_center)};
      if (i != j and pixDist > ellipses[i].m_minor and
          pixDist > ellipses[j].m_minor) {
        // j is not i, and they don't overlap
        // Are they close enough to each other to be counted as neighbors?
        auto const diff(positions[i] - positions[j]);
        if (diff.dot(diff) < bubbleSizeLimitSq) {
          ellipseNeighs[i].emplace_back(j);
        }
      }
    }
  }

  for (size_t i{0}; i < ellipseNeighs.size(); ++i) {
    if (ellipseNeighs[i].size() >= 5) {
      size_t neighsWithFourOrMoreSharedNeighs{0};
      for (auto const &neighIdx : ellipseNeighs[i]) {
        size_t sharedNeighs = 0;
        for (auto const &neighIdxInner : ellipseNeighs[i]) {
          if (neighIdx != neighIdxInner and
              std::find(std::begin(ellipseNeighs[neighIdx]),
                        std::end(ellipseNeighs[neighIdx]),
                        neighIdxInner) != std::end(ellipseNeighs[neighIdx]))
            sharedNeighs++;
        }
        if (sharedNeighs >= 4) {
          neighsWithFourOrMoreSharedNeighs++;
        }
      }
      if (neighsWithFourOrMoreSharedNeighs >= 5) {
        validEllipseIndices.emplace_back(i);
      }
    }
  }

  std::vector<hpm::Mark> marks;
  if (fitByDistance) {
    for (auto const &ellipseIndex : validEllipseIndices) {
      auto const e{ellipses[ellipseIndex]};
      // The ellipse center might be outside of picture frame
      int const row0{
          std::clamp(static_cast<int>(e.m_center.y), 0, imageMat.rows - 1)};
      int const col0{
          std::clamp(static_cast<int>(e.m_center.x), 0, imageMat.cols - 1)};
      int const rowUp{std::clamp(row0 - 1, 0, imageMat.rows - 1)};
      int const rowDown{std::clamp(row0 + 1, 0, imageMat.rows - 1)};
      int const colLeft{std::clamp(col0 - 1, 0, imageMat.cols - 1)};
      int const colRight{std::clamp(col0 + 1, 0, imageMat.cols - 1)};
      cv::Point3_<int> bgr0 =
          cv::Point3_<int>(imageMat.at<cv::Point3_<uint8_t>>(row0, col0));
      cv::Point3_<int> bgrLeft =
          cv::Point3_<int>(imageMat.at<cv::Point3_<uint8_t>>(row0, colLeft));
      cv::Point3_<int> bgrRight =
          cv::Point3_<int>(imageMat.at<cv::Point3_<uint8_t>>(row0, colRight));
      cv::Point3_<int> bgrUp =
          cv::Point3_<int>(imageMat.at<cv::Point3_<uint8_t>>(rowUp, col0));
      cv::Point3_<int> bgrDown =
          cv::Point3_<int>(imageMat.at<cv::Point3_<uint8_t>>(rowDown, col0));
      cv::Point3_<uint8_t> bgr = cv::Point3_<uint8_t>(
          (bgr0 + bgrLeft + bgrRight + bgrUp + bgrDown) / 5);

      if (not((bgr.x > 220 and bgr.y > 220 and bgr.z > 220) or
              (bgr.x < 35 and bgr.y < 35 and bgr.z < 35))) {
        // Not completely white or black..
        marks.emplace_back(e, bgr2hue(bgr));
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

  auto hueLessThan = [](hpm::Mark const &lhv, hpm::Mark const &rhv) -> bool {
    return lhv.m_hue < rhv.m_hue;
  };

  std::sort(marks.begin(), marks.end(), hueLessThan);

  if (marks.empty()) {
    return {};
  }

  std::vector<double> hueDistances(marks.size(), 0.0);
  for (size_t i{0}; i < marks.size() - 1; ++i) {
    hueDistances[i] = marks[i + 1].m_hue - marks[i].m_hue;
  }
  hueDistances.back() = marks[0].m_hue + (360.0 - marks.back().m_hue);
  std::array<std::vector<hpm::Mark>, 3> hueGroups;

  std::vector<size_t> const largestJumps = getLargest(hueDistances, 3);

  size_t groupIdx{0};
  for (size_t i{0}; i < marks.size(); ++i) {
    hueGroups[groupIdx % 3].emplace_back(marks[i]);
    if (i == largestJumps[0] or i == largestJumps[1] or i == largestJumps[2]) {
      // If the coming hue jump is one of the three largest ones, start
      // emplacing marks into the next hue group from now on
      groupIdx++;
    }
  }

  Marks result{};
  if (not(hueGroups[2].empty()) and not(hueGroups[0].empty()) and
      (360.0 - hueGroups[2].back().m_hue) < hueGroups[0][0].m_hue) {
    // hueGroup 2 is closer to 360 than hueGroup 0 is close to 0.
    result.m_red = hueGroups[2];
    result.m_green = hueGroups[0];
    result.m_blue = hueGroups[1];
  } else {
    result.m_red = hueGroups[0];
    result.m_green = hueGroups[1];
    result.m_blue = hueGroups[2];
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
