#include <hpm/marks.h++>

#include <limits>

using namespace hpm;

std::vector<Mark> Marks::getFlatCopy() const {
  std::vector<Mark> all{};
  all.reserve(size());
  all.insert(all.end(), m_red.begin(), m_red.end());
  all.insert(all.end(), m_green.begin(), m_green.end());
  all.insert(all.end(), m_blue.begin(), m_blue.end());
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

double Marks::identify(ProvidedMarkerPositions const &markPos,
                       double const focalLength,
                       PixelPosition const &imageCenter,
                       double const markerDiameter) {

  if (not(m_red.size() >= 2 and m_green.size() >= 2 and m_blue.size() >= 2)) {
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

  std::vector<double> expectedDists;
  expectedDists.reserve(NUMBER_OF_MARKERS * (NUMBER_OF_MARKERS - 1) / 2);
  for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
    for (size_t j{i + 1}; j < NUMBER_OF_MARKERS; ++j) {
      expectedDists.emplace_back(cv::norm(markPos.row(static_cast<int>(i)) -
                                          markPos.row(static_cast<int>(j))));
    }
  }

  auto const redIndexPairs{getIndexPairs(m_red.size())};
  auto const greenIndexPairs{getIndexPairs(m_green.size())};
  auto const blueIndexPairs{getIndexPairs(m_blue.size())};
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
      getPositions(m_red), getPositions(m_green), getPositions(m_blue)};

  // Find the sixtuple that produces the smallest squared error
  // but avoid calculating the full squared error for all sixtuples
  size_t bestSixtupleIdx = 0;
  double smallestErr = std::numeric_limits<double>::max();
  for (size_t sixtupleIdx{0}; sixtupleIdx < sixtuples.size(); sixtupleIdx++) {
    auto const sixtuple{sixtuples[sixtupleIdx]};
    size_t idx{0};
    double err{0.0};
    for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
      for (size_t j{i + 1}; j < NUMBER_OF_MARKERS; ++j) {
        auto dist{cv::norm(positions[i / 2][sixtuple[i]] -
                           positions[j / 2][sixtuple[j]])};
        auto diff{dist - expectedDists[idx]};
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

  m_red = {m_red[winnerSixtuple[0]], m_red[winnerSixtuple[1]]};
  m_green = {m_green[winnerSixtuple[2]], m_green[winnerSixtuple[3]]};
  m_blue = {m_blue[winnerSixtuple[4]], m_blue[winnerSixtuple[5]]};
  return smallestErr;
}
