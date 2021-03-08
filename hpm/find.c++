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

static Marks sortBySmallestJumps(std::vector<size_t> const &smallestJumps,
                                 std::vector<hpm::Mark> const &marks) {
  if (marks.size() == 1) {
    Marks result{};
    result.m_red.emplace_back(marks[0]);
    return result;
  }
  std::array<std::vector<hpm::Mark>, 2> hueGroups;
  std::vector<hpm::Mark> leastColorSortedGroup;
  size_t groupIdx{0};
  for (size_t i{0}; i < marks.size(); ++i) {
    if (i == smallestJumps[0] or i == smallestJumps[1]) {
      hueGroups[groupIdx % 3].emplace_back(marks[i]);
      hueGroups[groupIdx % 3].emplace_back(marks[(i + 1) % marks.size()]);
      groupIdx++;
      i++;
    } else {
      leastColorSortedGroup.emplace_back(marks[i]);
    }
  }
  Marks result{};
  std::vector<hpm::Mark> nonRed;
  if ((360.0 - hueGroups[1][0].m_hue) < hueGroups[0][0].m_hue) {
    // hueGroup 1 is closer to 360 than hueGroup 0 is close to 0.
    result.m_red = hueGroups[1];
    nonRed = hueGroups[0];
  } else {
    result.m_red = hueGroups[0];
    nonRed = hueGroups[1];
  }

  auto nonGreenNess = [](hpm::Mark const &mark) -> double {
    return std::abs(120.0 - mark.m_hue);
  };
  auto moreGreen = [nonGreenNess](hpm::Mark const &markLeft,
                                  hpm::Mark const &markRight) -> bool {
    return nonGreenNess(markLeft) < nonGreenNess(markRight);
  };

  hpm::Mark const &nonGreenestLeastColorSorted{
      *std::max_element(std::begin(leastColorSortedGroup),
                        std::end(leastColorSortedGroup), moreGreen)};
  hpm::Mark const &nonGreenestNonRed{
      *std::max_element(std::begin(nonRed), std::end(nonRed), moreGreen)};

  if (nonGreenNess(nonGreenestNonRed) <
      nonGreenNess(nonGreenestLeastColorSorted)) {
    result.m_green = nonRed;
    result.m_blue = leastColorSortedGroup;
  } else {
    result.m_green = leastColorSortedGroup;
    result.m_blue = nonRed;
  }
  return result;
}

static Marks sortByLargestJumps(std::vector<size_t> const &largestJumps,
                                std::vector<hpm::Mark> const &marks) {
  std::array<std::vector<hpm::Mark>, 3> hueGroups;
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

  // In case one marker ended up in the wrong group
  if (result.m_red.size() == 3 and result.m_blue.size() == 1) {
    result.m_blue.emplace_back(result.m_red[0]);
    result.m_red.erase(std::begin(result.m_red));
  } else if (result.m_blue.size() == 3 and result.m_green.size() == 1) {
    result.m_green.emplace_back(result.m_blue[0]);
    result.m_blue.erase(std::begin(result.m_blue));
  } else if (result.m_green.size() == 3 and result.m_red.size() == 1) {
    result.m_red.emplace_back(result.m_green[0]);
    result.m_green.erase(std::begin(result.m_green));
  }

  return result;
}

static auto
distanceGroupIndices(std::vector<hpm::CameraFramedPosition> const &positions,
                     std::vector<hpm::Ellipse> const &ellipses,
                     double const bubbleSizeLimit) -> std::vector<size_t> {
  std::vector<size_t> group{};

  double const limitSq{bubbleSizeLimit * bubbleSizeLimit};

  std::vector<std::vector<size_t>> ellipseNeighs(positions.size(),
                                                 std::vector<size_t>{});
  for (size_t i{0}; i < positions.size(); ++i) {
    for (size_t j{0}; j < positions.size(); ++j) {
      auto const pixDist{cv::norm(ellipses[i].m_center - ellipses[j].m_center)};
      if (i != j and pixDist > ellipses[i].m_minor and
          pixDist > ellipses[j].m_minor) {
        // j is not i, and they overlap less than half.
        // Are they close enough to each other to be counted as neighbors?
        auto const diff(positions[i] - positions[j]);
        if (diff.dot(diff) < limitSq) {
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
        group.emplace_back(i);
      }
    }
  }
  return group;
}

static auto getIndicesOfExtremes(std::vector<double> const &v,
                                 int howManyToSort, bool const less)
    -> std::vector<size_t> {
  struct CompLess {
    CompLess(const std::vector<double> &v) : _v(v) {}
    bool operator()(size_t a, size_t b) { return _v[a] < _v[b]; }
    const std::vector<double> &_v;
  };
  struct CompGreater {
    CompGreater(const std::vector<double> &v) : _v(v) {}
    bool operator()(size_t a, size_t b) { return _v[a] > _v[b]; }
    const std::vector<double> &_v;
  };
  std::vector<size_t> sortedIndices;
  sortedIndices.resize(v.size());
  for (size_t i = 0; i < sortedIndices.size(); ++i) {
    sortedIndices[i] = i;
  }
  if (less) {
    std::partial_sort(sortedIndices.begin(),
                      std::next(sortedIndices.begin(), howManyToSort),
                      sortedIndices.end(), CompLess(v));
  } else {
    std::partial_sort(sortedIndices.begin(),
                      std::next(sortedIndices.begin(), howManyToSort),
                      sortedIndices.end(), CompGreater(v));
  }
  sortedIndices.resize(static_cast<size_t>(howManyToSort));
  return sortedIndices;
}

auto find(cv::InputArray undistortedImage,
          hpm::ProvidedMarkerPositions const &markPos, double const focalLength,
          hpm::PixelPosition const &imageCenter, double const markerDiameter,
          bool showIntermediateImages, bool verbose, bool fitByDistance,
          PixelPosition const &expectedTopLeftestCenter) -> FindResult {
  Marks const marks{findMarks(undistortedImage, markPos, focalLength,
                              imageCenter, markerDiameter,
                              showIntermediateImages, verbose, fitByDistance,
                              expectedTopLeftestCenter)};
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

auto findMarks(cv::InputArray undistortedImage,
               hpm::ProvidedMarkerPositions const &markPos,
               double const focalLength, PixelPosition const &imageCenter,
               double const markerDiameter, bool showIntermediateImages,
               bool verbose, bool fitByDistance,
               PixelPosition const &expectedTopLeftestCenter) -> Marks {
  std::vector<hpm::Ellipse> const ellipses{ellipseDetect(
      undistortedImage, showIntermediateImages, expectedTopLeftestCenter)};
  if (ellipses.empty()) {
    return {};
  }

  cv::Mat imageMat{undistortedImage.getMat()};

  std::vector<hpm::CameraFramedPosition> positions;
  positions.reserve(ellipses.size());
  for (auto const &e : ellipses) {
    positions.emplace_back(
        e.toPosition(focalLength, imageCenter, markerDiameter));
  }

  std::vector<double> expectedDists;
  expectedDists.reserve(NUMBER_OF_MARKERS * (NUMBER_OF_MARKERS - 1) / 2);
  for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
    for (size_t j{i + 1}; j < NUMBER_OF_MARKERS; ++j) {
      double const dist = cv::norm(markPos.row(static_cast<int>(i)) -
                                   markPos.row(static_cast<int>(j)));
      expectedDists.emplace_back(dist);
    }
  }
  std::sort(std::begin(expectedDists), std::end(expectedDists));
  double constexpr BUBBLE_SIZE_MARGIN_FACTOR{1.7};
  double bubbleSizeLimit{expectedDists.back() * BUBBLE_SIZE_MARGIN_FACTOR};

  std::vector<size_t> validEllipseIndices{
      distanceGroupIndices(positions, ellipses, bubbleSizeLimit)};

  if (showIntermediateImages and fitByDistance) {
    std::vector<hpm::Ellipse> distanceGroupFiltered;
    for (auto const &ellipseIndex : validEllipseIndices) {
      distanceGroupFiltered.emplace_back(ellipses[ellipseIndex]);
    }
    cv::Mat cpy = imageMat.clone();
    for (auto const &ellipse : distanceGroupFiltered) {
      draw(cpy, ellipse, AQUA);
    }
    showImage(cpy, "distance-group-filtered-ones.png");
  }

  std::vector<std::array<size_t, 6>> candidateSixtuples;
  for (size_t a{0}; a < validEllipseIndices.size(); ++a) {
    for (size_t b{a + 1}; b < validEllipseIndices.size(); ++b) {
      for (size_t c{b + 1}; c < validEllipseIndices.size(); ++c) {
        for (size_t d{c + 1}; d < validEllipseIndices.size(); ++d) {
          for (size_t e{d + 1}; e < validEllipseIndices.size(); ++e) {
            for (size_t f{e + 1}; f < validEllipseIndices.size(); ++f) {
              candidateSixtuples.emplace_back(std::array<size_t, 6>{
                  validEllipseIndices[a], validEllipseIndices[b],
                  validEllipseIndices[c], validEllipseIndices[d],
                  validEllipseIndices[e], validEllipseIndices[f]});
            }
          }
        }
      }
    }
  }

  size_t bestSixtupleIdx{0UL};
  if (candidateSixtuples.size() > 1) {
    double bestErr{std::numeric_limits<double>::max()};
    for (size_t i{0}; i < candidateSixtuples.size(); ++i) {
      auto const sixtuple{candidateSixtuples[i]};
      std::vector<double> dists;
      for (size_t j{0}; j < NUMBER_OF_MARKERS; ++j) {
        for (size_t k{j + 1}; k < NUMBER_OF_MARKERS; ++k) {
          double const dist =
              cv::norm(positions[sixtuple[j]] - positions[sixtuple[k]]);
          dists.emplace_back(dist);
        }
      }
      std::sort(std::begin(dists), std::end(dists));
      double err{0.0};
      for (size_t distIdx{0}; distIdx < dists.size(); distIdx++) {
        err += pow(dists[distIdx] - expectedDists[distIdx], 2);
      }
      if (err < bestErr) {
        bestSixtupleIdx = i;
        bestErr = err;
      }
    }
  }

  std::vector<hpm::Mark> marks;
  if (fitByDistance and not candidateSixtuples.empty()) {
    for (auto const &ellipseIndex : candidateSixtuples[bestSixtupleIdx]) {
      // clamp because the ellipse center might be outside of picture frame
      auto const e{ellipses[ellipseIndex]};
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

      marks.emplace_back(e, bgr2hue(bgr));
      // std::cout << bgr2hue(bgr) * CV_PI / 180.0 << ", 0, ";
    }
    if (showIntermediateImages) {
      cv::Mat cpy = imageMat.clone();
      for (auto const &mark : marks) {
        draw(cpy, mark, AQUA);
      }
      showImage(cpy, "total-distance-filtered-ones.png");
    }
  } else {
    for (auto const &e : ellipses) {
      marks.emplace_back(e, 0.0);
    }
  }

  auto hueLessThan = [](hpm::Mark const &lhv, hpm::Mark const &rhv) -> bool {
    return lhv.m_hue < rhv.m_hue;
  };

  std::sort(marks.begin(), marks.end(), hueLessThan);

  if (marks.size() < 3) {
    Marks result{};
    if (marks.size() > 0) {
      result.m_red.emplace_back(marks[0]);
    }
    return result;
  }

  std::vector<double> hueDistances(marks.size(), 0.0);
  for (size_t i{0}; i < marks.size() - 1; ++i) {
    hueDistances[i] = marks[i + 1].m_hue - marks[i].m_hue;
  }
  hueDistances.back() = marks[0].m_hue + (360.0 - marks.back().m_hue);

  Marks result{};
  std::vector<size_t> const smallestJumps =
      getIndicesOfExtremes(hueDistances, 3, true);
  if (smallestJumps[0] != (smallestJumps[1] - 1) and
      smallestJumps[1] != (smallestJumps[0] - 1) and
      hueDistances[smallestJumps[2]] > 10.0) {
    result = sortBySmallestJumps(smallestJumps, marks);
  } else {
    std::vector<size_t> const largestJumps =
        getIndicesOfExtremes(hueDistances, 3, false);
    result = sortByLargestJumps(largestJumps, marks);
  }

  if (showIntermediateImages) {
    showImage(imageWith(undistortedImage, result),
              "color-categorized-ones.png");
  }
  if (fitByDistance) {
    result.identify(markPos, focalLength, imageCenter, markerDiameter);
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
