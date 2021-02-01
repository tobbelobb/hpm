#include <hpm/marks.h++>

#include <limits>

using namespace hpm;

std::vector<Mark> Marks::getFlatCopy() const {
  std::vector<Mark> all{};
  all.reserve(size());
  all.insert(all.end(), red.begin(), red.end());
  all.insert(all.end(), green.begin(), green.end());
  all.insert(all.end(), blue.begin(), blue.end());
  return all;
}

static std::vector<std::pair<size_t, size_t>> getIndexPairs(size_t sz) {
  std::vector<std::pair<size_t, size_t>> indexPairs{};
  for (size_t i{0}; i < sz; ++i) {
    for (size_t j{0}; j < sz; ++j) {
      if (i != j) {
        indexPairs.emplace_back(i, j);
      }
    }
  }
  return indexPairs;
}

double Marks::fit(ProvidedMarkerPositions const &markPos,
                  double const focalLength, PixelPosition const &imageCenter,
                  double const markerDiameter) {

  if (not(red.size() >= 2 and green.size() >= 2 and blue.size() >= 2)) {
    return std::numeric_limits<double>::max();
  }

  auto getPositions =
      [&](std::vector<Mark> const &marks) -> std::vector<CameraFramedPosition> {
    std::vector<CameraFramedPosition> positions{};
    for (auto const &mark : marks) {
      positions.emplace_back(
          mark.toPosition(focalLength, imageCenter, markerDiameter));
    }
    return positions;
  };

  size_t constexpr NUMBER_OF_DISTANCES{NUMBER_OF_MARKERS *
                                       (NUMBER_OF_MARKERS - 1) / 2};
  cv::Vec<double, NUMBER_OF_DISTANCES> expectedDistances{};
  {
    int idx = 0;
    for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
      for (size_t j{i + 1}; j < NUMBER_OF_MARKERS; ++j) {
        expectedDistances[idx] = cv::norm(markPos.row(static_cast<int>(i)) -
                                          markPos.row(static_cast<int>(j)));
        idx++;
      }
    }
  }

  auto const redIndexPairs{getIndexPairs(red.size())};
  auto const greenIndexPairs{getIndexPairs(green.size())};
  auto const blueIndexPairs{getIndexPairs(blue.size())};
  // These are all the possible ordered combinations of marks
  // expressed as indices into the current red/green/blue vectors or marks.
  std::vector<std::array<size_t, 6>> sixtuples{};
  for (auto const &redPair : redIndexPairs) {
    for (auto const &greenPair : greenIndexPairs) {
      for (auto const &bluePair : blueIndexPairs) {
        sixtuples.emplace_back(std::array<size_t, 6>{
            redPair.first, redPair.second, greenPair.first, greenPair.second,
            bluePair.first, bluePair.second});
      }
    }
  }

  std::array<std::vector<CameraFramedPosition>, 3> const positions = {
      getPositions(red), getPositions(green), getPositions(blue)};

  // Find the sixtuple that produces the smallest squared error
  // but avoid calculating the full squared error for all sixtuples
  size_t bestSixtupleIdx = 0;
  double smallestErr = std::numeric_limits<double>::max();
  for (size_t sixtupleIdx{0}; sixtupleIdx < sixtuples.size(); sixtupleIdx++) {
    auto const sixtuple{sixtuples[sixtupleIdx]};
    int idx{0};
    double err{0.0};
    for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
      for (size_t j{i + 1}; j < NUMBER_OF_MARKERS; ++j) {
        auto dist{cv::norm(positions[i / 2][sixtuple[i]] -
                           positions[j / 2][sixtuple[j]])};
        auto diff{dist - expectedDistances[idx]};
        idx++;
        err = err + diff * diff;
        if (err > smallestErr) { // short out of hot loop
          i = NUMBER_OF_MARKERS;
          j = NUMBER_OF_MARKERS;
        }
      }
    }
    if (err < smallestErr) {
      smallestErr = err;
      bestSixtupleIdx = sixtupleIdx;
    }
  }

  std::array<size_t, 6> const winnerSixtuple = sixtuples[bestSixtupleIdx];

  red = {red[winnerSixtuple[0]], red[winnerSixtuple[1]]};
  green = {green[winnerSixtuple[2]], green[winnerSixtuple[3]]};
  blue = {blue[winnerSixtuple[4]], blue[winnerSixtuple[5]]};
  return smallestErr;
}
