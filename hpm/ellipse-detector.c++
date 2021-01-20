#include <hpm/ed/EDLib.h++>

#include <hpm/detection-result.h++>
#include <hpm/ellipse-detector.h++>
#include <hpm/util.h++>

using namespace hpm;

struct HuedKeyPoint {
  hpm::KeyPoint keyPoint;
  uint8_t hue;
};

static auto getBigEllipses(EDCircles const &edCircles, double sizeThreshold)
    -> std::vector<hpm::KeyPoint> {
  std::vector<hpm::KeyPoint> bigEllipses{};
  for (auto const &circle : edCircles.getCirclesRef()) {
    if (circle.r > sizeThreshold) {
      bigEllipses.emplace_back(circle);
    }
  }
  for (auto const &ellipse : edCircles.getEllipsesRef()) {
    if (ellipse.axes.width > sizeThreshold and
        ellipse.axes.height > sizeThreshold) {
      bigEllipses.emplace_back(ellipse);
    }
  }
  return bigEllipses;
}

auto ellipseDetect(cv::InputArray image, bool showIntermediateImages)
    -> hpm::DetectionResult {
  cv::Mat imageMat{image.getMat()};
  EDColor const edColor{
      imageMat,
      {.gradThresh = 27, // lower gradThresh finds more ellipses, both true and
                         // false positives
       .anchorThresh = 4,
       .blurSize = 1.5,
       .filterSegments = true}};
  if (showIntermediateImages) {
    showImage(edColor.getEdgeImage(), "edgeImage.png");
  }

  EDCircles const edCircles{edColor};
  if (showIntermediateImages) {
    showImage(edCircles.drawResult(imageMat, ImageStyle::BOTH),
              "edCircles.png");
  }

  // Size of a marker must be at least 1/100 of the image width
  double const sizeThresholdNominator{static_cast<double>(imageMat.cols)};
  double constexpr SIZE_THRESHOLD_DENOMINATOR{100.0};
  double const sizeThreshold{sizeThresholdNominator /
                             SIZE_THRESHOLD_DENOMINATOR};
  std::vector<hpm::KeyPoint> bigEllipses{
      getBigEllipses(edCircles, sizeThreshold)};

  if (showIntermediateImages) {
    cv::Mat cpy = imageMat.clone();
    drawKeyPoints(cpy, bigEllipses, cv::Scalar(255, 255, 0));
    showImage(cpy, "big-ellipses.png");
  }

  std::vector<hpm::KeyPoint> centerPointingEllipses;
  for (auto const &e : bigEllipses) {
    PixelPosition const center{static_cast<double>(imageMat.cols) / 2.0,
                               static_cast<double>(imageMat.rows) / 2.0};
    PixelPosition const distCoord = e.m_center - center;
    double const shouldAngle = atan(distCoord.y / distCoord.x);
    double const dist =
        sqrt(distCoord.x * distCoord.x + distCoord.y * distCoord.y);
    double const maxDist{0.5 * static_cast<double>(imageMat.rows)};
    if (dist < maxDist and e.m_major == e.m_minor) {
      // a circle near middle of image
      centerPointingEllipses.emplace_back(e);
    } else if (std::abs(shouldAngle - e.m_rot) < (17.0 * M_PI / 180.0) and
               e.m_major != e.m_minor) {
      // a center pointing ellipse
      centerPointingEllipses.emplace_back(e);
    }
  }
  if (showIntermediateImages) {
    cv::Mat cpy = imageMat.clone();
    drawKeyPoints(cpy, centerPointingEllipses, cv::Scalar(255, 255, 0));
    showImage(cpy, "center-pointing-ellipses.png");
  }

  if (centerPointingEllipses.empty()) {
    return {};
  }

  // TODO: Pick out hue channel only
  cv::Mat hsv{};
  cv::cvtColor(imageMat, hsv, cv::COLOR_BGR2HSV);
  cv::Mat hue = getHueChannelCopy(hsv);

  std::vector<HuedKeyPoint> huedEllipses;
  for (auto const &ellipse : centerPointingEllipses) {
    huedEllipses.emplace_back(ellipse,
                              (hue.at<uint8_t>(ellipse.m_center) + 30) % 180);
  }

  std::sort(
      huedEllipses.begin(), huedEllipses.end(),
      [&](auto const &lhv, auto const &rhv) { return lhv.hue < rhv.hue; });

  double const min = static_cast<double>(huedEllipses[0].hue);
  double const max = static_cast<double>(huedEllipses.back().hue);
  double const diff = max - min;
  double redMid = min + diff / 6.0;
  double greenMid = huedEllipses[huedEllipses.size() / 2].hue;
  double blueMid = max - diff / 6.0;

  DetectionResult result{};
  for (auto const &he : huedEllipses) {
    double colorDistanceRed = std::abs(static_cast<double>(he.hue) - redMid);
    double colorDistanceGreen =
        std::abs(static_cast<double>(he.hue) - greenMid);
    double colorDistanceBlue = std::abs(static_cast<double>(he.hue) - blueMid);
    if (colorDistanceRed < colorDistanceGreen and
        colorDistanceRed < colorDistanceBlue) {
      result.red.emplace_back(he.keyPoint);
    } else if (colorDistanceGreen < colorDistanceRed and
               colorDistanceGreen < colorDistanceBlue) {
      result.green.emplace_back(he.keyPoint);
    } else {
      result.blue.emplace_back(he.keyPoint);
    }
  }

  return result;
}

auto ellipseToPosition(hpm::KeyPoint const &ellipse, double focalLength,
                       hpm::PixelPosition const &imageCenter,
                       double markerDiameter) -> hpm::CameraFramedPosition {
  PixelPosition const fromCenter = ellipse.m_center - imageCenter;
  double const lengthFromOrigin = cv::norm(fromCenter);
  PixelPosition const dirToOrigin = lengthFromOrigin == 0.0
                                        ? PixelPosition{1.0, 0}
                                        : -fromCenter / lengthFromOrigin;
  double const majorAxis = ellipse.m_major;
  double const semiMajorAxis = majorAxis / 2.0;
  PixelPosition const closestPoint = fromCenter + semiMajorAxis * dirToOrigin;
  PixelPosition const farthestPoint = fromCenter - semiMajorAxis * dirToOrigin;
  double const largestAng = atan(cv::norm(farthestPoint) / focalLength);
  double smallestAng = atan(cv::norm(closestPoint) / focalLength);
  if (lengthFromOrigin < semiMajorAxis) {
    smallestAng = -smallestAng;
  }
  // facing disc's midpoint ang
  double const alpha = std::midpoint(smallestAng, largestAng);
  // facing disc's angular radius seen from the pinhole
  double const gamma1 = std::midpoint(largestAng, -smallestAng);

  // We know that
  //   gamma1 = asin(r/d),
  // where r is markerDiameter/2,
  // and d is the marker's total distance from the pinhole
  //
  double const rot = atan2(fromCenter.y, fromCenter.x);

  double const d = (markerDiameter / 2.0) / sin(gamma1);
  double const dxy = sin(alpha) * d;
  double const z = cos(alpha) * d;
  double const x = dxy * cos(rot);
  double const y = dxy * sin(rot);

  return {x, y, z};
}
