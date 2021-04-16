#include <hpm/ellipse-detector.h++>
#include <hpm/find.h++>
#include <hpm/util.h++>

#include <hpm/warnings-disabler.h++>
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
    -> std::vector<std::array<size_t, NUMBER_OF_MARKERS>> {
  if (indices.size() < NUMBER_OF_MARKERS) {
    return {};
  }

  std::vector<std::array<size_t, NUMBER_OF_MARKERS>> sixtuples;
  for (size_t a{0}; a < indices.size(); ++a) {
    for (size_t b{a + 1}; b < indices.size(); ++b) {
      for (size_t c{b + 1}; c < indices.size(); ++c) {
        for (size_t d{c + 1}; d < indices.size(); ++d) {
          for (size_t e{d + 1}; e < indices.size(); ++e) {
            for (size_t f{e + 1}; f < indices.size(); ++f) {
              std::array<size_t, NUMBER_OF_MARKERS> const candidate{
                  indices[a], indices[b], indices[c],
                  indices[d], indices[e], indices[f]};
              sixtuples.emplace_back(candidate);
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
    if (ellipseNeighs[i].size() >= (NUMBER_OF_MARKERS - 1)) {
      size_t neighsWithFourOrMoreSharedNeighs{0};
      for (auto const &neighIdx : ellipseNeighs[i]) {
        size_t sharedNeighs = 0;
        for (auto const &neighIdxInner : ellipseNeighs[i]) {
          if (neighIdx != neighIdxInner and
              std::find(std::begin(ellipseNeighs[neighIdx]),
                        std::end(ellipseNeighs[neighIdx]),
                        neighIdxInner) != std::end(ellipseNeighs[neighIdx])) {
            sharedNeighs++;
          }
        }
        if (sharedNeighs >= (NUMBER_OF_MARKERS - 2)) {
          neighsWithFourOrMoreSharedNeighs++;
        }
      }
      if (neighsWithFourOrMoreSharedNeighs >= (NUMBER_OF_MARKERS - 1)) {
        group.emplace_back(i);
      }
    }
  }
  return group;
}

auto findMarks(FinderImage const &image, MarkerParams const &markerParams,
               FinderConfig const &config,
               CameraFramedPosition const &expectedNormalDirection,
               bool tryHard) -> std::vector<hpm::Ellipse> {
  std::vector<hpm::Ellipse> const ellipses{
      ellipseDetect(image.m_mat, config.m_showIntermediateImages,
                    markerParams.m_topLeftMarkerCenter)};
  if (ellipses.empty()) {
    return {};
  }

  std::vector<hpm::CameraFramedPosition> positions;
  positions.reserve(ellipses.size());
  for (auto const &e : ellipses) {
    positions.emplace_back(toPosition(
        e, markerParams.m_diameter, image.m_focalLength, image.m_center,
        markerParams.m_type, expectedNormalDirection));
  }

  std::vector<double> expectedDists;
  expectedDists.reserve(NUMBER_OF_MARKERS * (NUMBER_OF_MARKERS - 1) / 2);
  for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
    for (size_t j{i + 1}; j < NUMBER_OF_MARKERS; ++j) {
      double const dist = cv::norm(
          markerParams.m_providedMarkerPositions.row(static_cast<int>(i)) -
          markerParams.m_providedMarkerPositions.row(static_cast<int>(j)));
      expectedDists.emplace_back(dist);
    }
  }
  std::sort(std::begin(expectedDists), std::end(expectedDists));
  double constexpr BUBBLE_SIZE_MARGIN_FACTOR{2.0};
  double bubbleSizeLimit{expectedDists.back() * BUBBLE_SIZE_MARGIN_FACTOR};

  std::vector<size_t> validEllipseIndices{
      distanceGroupIndices(positions, ellipses, bubbleSizeLimit)};
  if (tryHard and validEllipseIndices.empty()) {
    for (size_t i{0}; i < ellipses.size(); ++i) {
      validEllipseIndices.emplace_back(i);
    }
  }
  size_t doublettes{0};
  if (tryHard and validEllipseIndices.size() == NUMBER_OF_MARKERS - 1) {
    // Insert a doublette as a workaround.
    // Should be sorted out by later stages.
    validEllipseIndices.emplace_back(0);
    doublettes = 1;
  }

  if (config.m_showIntermediateImages and config.m_fitByDistance) {
    std::vector<hpm::Ellipse> distanceGroupFiltered;
    distanceGroupFiltered.reserve(validEllipseIndices.size());
    for (auto const &ellipseIndex : validEllipseIndices) {
      distanceGroupFiltered.emplace_back(ellipses[ellipseIndex]);
    }
    cv::Mat cpy = image.m_mat.clone();
    for (auto const &ellipse : distanceGroupFiltered) {
      draw(cpy, ellipse, AQUA);
    }
    std::string imageName{"distance-group-filtered-ones.png"};
    if (tryHard) {
      imageName = "distance-group-filtered-ones-try-hard.png";
    }
    showImage(cpy, imageName);
  }

  std::vector<std::array<size_t, NUMBER_OF_MARKERS>> const candidateSixtuples{
      extractSixtuples(validEllipseIndices)};

  size_t bestSixtupleIdx{0UL};
  if (candidateSixtuples.size() > 1) {
    std::vector<std::vector<double>> distvs;
    distvs.reserve(candidateSixtuples.size());
    for (auto const &sixtuple : candidateSixtuples) {
      std::vector<double> distv;
      distv.reserve((NUMBER_OF_MARKERS * (NUMBER_OF_MARKERS - 1)) / 2);
      for (size_t j{0}; j < NUMBER_OF_MARKERS; ++j) {
        for (size_t k{j + 1}; k < NUMBER_OF_MARKERS; ++k) {
          double const dist = cv::norm(positions[sixtuple[j]] - // NOLINT
                                       positions[sixtuple[k]]); // NOLINT
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
    errs.reserve(candidateSixtuples.size());
    for (auto const &distv : distvs) {
      errs.emplace_back(distErr(distv));
    }

    double constexpr OVERLAP_TAX{10000.0};
    for (size_t cand{0}; cand < candidateSixtuples.size(); ++cand) {
      auto const candidate{candidateSixtuples[cand]};
      for (size_t i{0}; i < candidate.size() - 1; ++i) {
        for (size_t j{i + 1}; j < candidate.size(); ++j) {
          auto const pixDist{cv::norm(ellipses[candidate[i]].m_center -
                                      ellipses[candidate[j]].m_center)};
          if (pixDist < ellipses[candidate[i]].m_minor / 2.0 +
                            ellipses[candidate[j]].m_minor / 2.0) {
            errs[cand] += OVERLAP_TAX;
          }
        }
      }
    }

    bestSixtupleIdx = static_cast<size_t>(std::distance(
        std::begin(errs), std::min_element(std::begin(errs), std::end(errs))));
  }

  std::vector<hpm::Ellipse> marks;
  if (config.m_fitByDistance and not candidateSixtuples.empty()) {
    for (auto const &ellipseIndex : candidateSixtuples[bestSixtupleIdx]) {
      marks.emplace_back(ellipses[ellipseIndex]);
    }
    if (config.m_showIntermediateImages) {
      cv::Mat cpy = image.m_mat.clone();
      for (auto const &mark : marks) {
        draw(cpy, mark, AQUA);
      }

      std::string imageName{"total-distance-filtered-ones.png"};
      if (tryHard) {
        imageName = "total-distance-filtered-ones-try-hard.png";
      }
      showImage(cpy, imageName);
    }
  } else {
    for (auto const &e : ellipses) {
      marks.emplace_back(e);
    }
  }

  if (config.m_fitByDistance and marks.size() == NUMBER_OF_MARKERS) {
    identify(marks, markerParams.m_diameter,
             markerParams.m_providedMarkerPositions, image.m_focalLength,
             image.m_center, markerParams.m_type, tryHard);
  }

  if (config.m_verbose) {
    std::cout << "Found " << marks.size() - doublettes << " markers\n";
  }
  return marks;
}

auto findIndividualMarkerPositions(
    std::vector<hpm::Ellipse> const &marks, double const knownMarkerDiameter,
    double const focalLength, PixelPosition const &imageCenter,
    MarkerType markerType, CameraFramedPosition const &expectedNormalDirection)
    -> std::vector<CameraFramedPosition> {
  std::vector<CameraFramedPosition> positions{};
  positions.reserve(marks.size());
  for (auto const &detected : marks) {
    positions.emplace_back(toPosition(detected, knownMarkerDiameter,
                                      focalLength, imageCenter, markerType,
                                      expectedNormalDirection));
  }

  return positions;
}
