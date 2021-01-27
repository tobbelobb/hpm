#include <hpm/detection-result.h++>
#include <hpm/ellipse-detector.h++>

using namespace hpm;

auto hpm::zFromSemiMinor(double markerR, double f, double semiMinor) -> double {
  double const rSmall = markerR * f / sqrt(semiMinor * semiMinor + f * f);
  double const thetaZ = atan(semiMinor / f);
  return rSmall * f / semiMinor + markerR * sin(thetaZ);
}

auto hpm::centerRayFromZ(double c, double markerR, double z) -> double {
  return c * (z * z - markerR * markerR) / (z * z);
}

auto hpm::KeyPoint::getCenterRay(double const markerR, double const f,
                                 PixelPosition const &imageCenter) const
    -> PixelPosition {
  double const z = zFromSemiMinor(markerR, f, m_minor / 2);
  PixelPosition const imageCenterToEllipseCenter = m_center - imageCenter;
  double const c = cv::norm(imageCenterToEllipseCenter);
  double const centerRay = centerRayFromZ(c, markerR, z);
  return imageCenter + centerRay * imageCenterToEllipseCenter / c;
}

hpm::KeyPoint::KeyPoint(mEllipse const &ellipse) : m_center(ellipse.center) {
  if (ellipse.axes.width >= ellipse.axes.height) {
    m_major = 2.0 * ellipse.axes.width;
    m_minor = 2.0 * ellipse.axes.height;
    m_rot = ellipse.theta;
  } else {
    m_major = 2.0 * ellipse.axes.height;
    m_minor = 2.0 * ellipse.axes.width;
    if (ellipse.theta > 0.0) {
      m_rot = ellipse.theta - M_PI / 2.0;
    } else {
      m_rot = ellipse.theta + M_PI / 2.0;
    }
  }
}

std::vector<hpm::KeyPoint> DetectionResult::getFlatCopy() const {
  std::vector<hpm::KeyPoint> all{};
  all.reserve(size());
  all.insert(all.end(), red.begin(), red.end());
  all.insert(all.end(), green.begin(), green.end());
  all.insert(all.end(), blue.begin(), blue.end());
  return all;
}

void DetectionResult::filterByDistance(ProvidedMarkerPositions const &markPos,
                                       double const focalLength,
                                       PixelPosition const &imageCenter,
                                       double const markerDiameter) {
  auto filterSingleColor = [&](std::vector<KeyPoint> &marksOfOneColor,
                               double expectedDistance) {
    size_t const sz{marksOfOneColor.size()};
    if (sz > 2) {

      std::vector<CameraFramedPosition> allPositions{};
      for (auto const &mark : marksOfOneColor) {
        allPositions.emplace_back(
            ellipseToPosition(mark, focalLength, imageCenter, markerDiameter));
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
