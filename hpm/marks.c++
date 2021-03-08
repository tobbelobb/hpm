#include <hpm/marks.h++>

#include <algorithm>
#include <limits>

using namespace hpm;

static inline auto signed2DCross(PixelPosition const &v0,
                                 PixelPosition const &v1,
                                 PixelPosition const &v2) {
  return (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
}

static inline auto isRight(PixelPosition const &v0, PixelPosition const &v1,
                           PixelPosition const &v2) -> bool {
  return signed2DCross(v0, v1, v2) <= 0.0;
}

static void fanSort(std::vector<hpm::Mark> &fan) {
  const auto &pivot = fan[0];
  std::sort(std::next(fan.begin()), fan.end(),
            [&pivot](hpm::Mark const &lhs, hpm::Mark const &rhs) -> bool {
              return isRight(pivot.getCenter(), lhs.getCenter(),
                             rhs.getCenter());
            });
}

double Marks::identify(ProvidedMarkerPositions const &markPos,
                       double const focalLength,
                       PixelPosition const &imageCenter,
                       double const markerDiameter) {

  if (not(size() >= NUMBER_OF_MARKERS)) {
    return std::numeric_limits<double>::max();
  }

  fanSort(m_marks);

  std::vector<double> expectedDists;
  expectedDists.reserve(NUMBER_OF_MARKERS);
  for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
    expectedDists.emplace_back(
        cv::norm(markPos.row(static_cast<int>(i)) -
                 markPos.row(static_cast<int>((i + 1) % NUMBER_OF_MARKERS))));
  }

  std::vector<CameraFramedPosition> positions{};
  for (auto const &mark : m_marks) {
    positions.emplace_back(
        mark.toPosition(focalLength, imageCenter, markerDiameter));
  }
  std::vector<double> foundDists;
  foundDists.reserve(NUMBER_OF_MARKERS);
  for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
    foundDists.emplace_back(
        cv::norm(positions[i] - positions[(i + 1) % NUMBER_OF_MARKERS]));
  }

  std::vector<double> errs;
  errs.reserve(NUMBER_OF_MARKERS);
  for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
    double err{0.0};
    for (size_t j{0}; j < NUMBER_OF_MARKERS; ++j) {
      double const diff{foundDists[(i + j) % NUMBER_OF_MARKERS] -
                        expectedDists[j]};
      err += diff * diff;
    }
    errs.emplace_back(err);
  }

  auto const bestErrIdx{std::distance(
      std::begin(errs), std::min_element(std::begin(errs), std::end(errs)))};
  std::rotate(std::begin(m_marks), std::begin(m_marks) + bestErrIdx,
              std::end(m_marks));

  return errs[static_cast<size_t>(bestErrIdx)];
}
