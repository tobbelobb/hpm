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
