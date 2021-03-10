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

static auto extractSixtuples(std::vector<size_t> indices)
    -> std::vector<std::array<size_t, 6>> {
  if (indices.size() < 6) {
    return {};
  }

  std::vector<std::array<size_t, 6>> sixtuples;
  for (size_t a{0}; a < indices.size(); ++a) {
    for (size_t b{a + 1}; b < indices.size(); ++b) {
      for (size_t c{b + 1}; c < indices.size(); ++c) {
        for (size_t d{c + 1}; d < indices.size(); ++d) {
          for (size_t e{d + 1}; e < indices.size(); ++e) {
            for (size_t f{e + 1}; f < indices.size(); ++f) {
              sixtuples.emplace_back(
                  std::array<size_t, 6>{indices[a], indices[b], indices[c],
                                        indices[d], indices[e], indices[f]});
            }
          }
        }
      }
    }
  }
  return sixtuples;
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

  std::vector<std::array<size_t, 6>> const candidateSixtuples{
      extractSixtuples(validEllipseIndices)};

  size_t bestSixtupleIdx{0UL};
  if (candidateSixtuples.size() > 1) {
    std::vector<std::vector<double>> distvs;
    for (auto const &sixtuple : candidateSixtuples) {
      std::vector<double> distv;
      for (size_t j{0}; j < NUMBER_OF_MARKERS; ++j) {
        for (size_t k{j + 1}; k < NUMBER_OF_MARKERS; ++k) {
          double const dist =
              cv::norm(positions[sixtuple[j]] - positions[sixtuple[k]]);
          distv.emplace_back(dist);
        }
      }
      std::sort(std::begin(distv), std::end(distv));
      distvs.emplace_back(distv);
    }

    auto distErr = [&expectedDists](std::vector<double> const &distv) {
      double err{0.0};
      for (size_t i{0}; i < distv.size(); ++i) {
        double const diff{distv[i] - expectedDists[i]};
        err += diff * diff;
      }
      return err;
    };

    std::vector<double> errs;
    errs.reserve(distvs.size());
    for (auto const &distv : distvs) {
      errs.emplace_back(distErr(distv));
    }

    bestSixtupleIdx = static_cast<size_t>(std::distance(
        std::begin(errs), std::min_element(std::begin(errs), std::end(errs))));
  }

  std::vector<hpm::Mark> marks;
  if (fitByDistance and not candidateSixtuples.empty()) {
    for (auto const &ellipseIndex : candidateSixtuples[bestSixtupleIdx]) {
      marks.emplace_back(ellipses[ellipseIndex]);
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
      marks.emplace_back(e);
    }
  }

  Marks result{std::move(marks)};
  if (fitByDistance and result.size() == NUMBER_OF_MARKERS) {
    result.identify(markPos, focalLength, imageCenter, markerDiameter);
  }
  if (verbose) {
    std::cout << "Found " << result.size() << " markers\n";
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
  for (auto const &detected : marks.m_marks) {
    positions.emplace_back(
        detected.toPosition(focalLength, imageCenter, knownMarkerDiameter));
  }

  return positions;
}
