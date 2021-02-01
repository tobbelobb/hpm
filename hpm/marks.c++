#include <hpm/marks.h++>

using namespace hpm;

std::vector<Mark> Marks::getFlatCopy() const {
  std::vector<Mark> all{};
  all.reserve(size());
  all.insert(all.end(), red.begin(), red.end());
  all.insert(all.end(), green.begin(), green.end());
  all.insert(all.end(), blue.begin(), blue.end());
  return all;
}

void Marks::filterByDistance(ProvidedMarkerPositions const &markPos,
                             double const focalLength,
                             PixelPosition const &imageCenter,
                             double const markerDiameter) {
  auto filterSingleColor = [&](std::vector<Mark> &marksOfOneColor,
                               double expectedDistance) {
    size_t const sz{marksOfOneColor.size()};
    if (sz > 2) {

      std::vector<CameraFramedPosition> allPositions{};
      for (auto const &mark : marksOfOneColor) {
        allPositions.emplace_back(
            mark.toPosition(focalLength, imageCenter, markerDiameter));
      }

      std::vector<std::pair<size_t, size_t>> allPairs{};
      for (size_t i{0}; i < sz; ++i) {
        for (size_t j{i + 1}; j < sz; ++j) {
          allPairs.emplace_back(i, j);
        }
      }

      std::vector<double> allDistances{};
      for (auto const &pair : allPairs) {
        allDistances.emplace_back(
            cv::norm(allPositions[pair.first] - allPositions[pair.second]));
      }

      auto const winnerPair = allPairs[static_cast<size_t>(std::distance(
          allDistances.begin(),
          std::min_element(
              allDistances.begin(), allDistances.end(),
              [expectedDistance](double distanceLeft, double distanceRight) {
                return abs(distanceLeft - expectedDistance) <
                       abs(distanceRight - expectedDistance);
              })))];

      auto const first{marksOfOneColor[winnerPair.first]};
      auto const second{marksOfOneColor[winnerPair.second]};
      marksOfOneColor.clear();
      marksOfOneColor.push_back(first);
      marksOfOneColor.push_back(second);
    }
  };

  double const redDistance = cv::norm(markPos.row(0) - markPos.row(1));
  double const greenDistance = cv::norm(markPos.row(2) - markPos.row(3));
  double const blueDistance = cv::norm(markPos.row(4) - markPos.row(5));
  filterSingleColor(red, redDistance);
  filterSingleColor(green, greenDistance);
  filterSingleColor(blue, blueDistance);
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

void Marks::filterAndSortByDistance(ProvidedMarkerPositions const &markPos,
                                    double const focalLength,
                                    PixelPosition const &imageCenter,
                                    double const markerDiameter) {

  if (not(red.size() >= 2 and green.size() >= 2 and blue.size() >= 2)) {
    return;
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

  std::vector<cv::Vec<double, NUMBER_OF_DISTANCES>> distancesVec{};
  for (auto const &sixtuple : sixtuples) {
    cv::Vec<double, NUMBER_OF_DISTANCES> distances{};
    {
      int idx = 0;
      for (size_t i{0}; i < NUMBER_OF_MARKERS; ++i) {
        for (size_t j{i + 1}; j < NUMBER_OF_MARKERS; ++j) {
          distances[idx] = cv::norm(positions[i / 2][sixtuple[i]] -
                                    positions[j / 2][sixtuple[j]]);
          idx++;
        }
      }
    }
    distancesVec.emplace_back(distances);
  }

  std::vector<double> errs{};
  errs.reserve(sixtuples.size());
  for (auto const &distances : distancesVec) {
    errs.emplace_back(cv::norm(distances - expectedDistances));
  }

  std::array<size_t, 6> const winnerSixtuple = sixtuples[static_cast<size_t>(
      std::distance(errs.begin(), std::min_element(errs.begin(), errs.end())))];

  red = {red[winnerSixtuple[0]], red[winnerSixtuple[1]]};
  green = {green[winnerSixtuple[2]], green[winnerSixtuple[3]]};
  blue = {blue[winnerSixtuple[4]], blue[winnerSixtuple[5]]};
}
