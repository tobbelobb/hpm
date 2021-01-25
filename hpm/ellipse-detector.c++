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
      {.gradThresh = 10, // lower gradThresh finds more ellipses, both true and
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

  // Size of a marker must be at least 1/200 of the image width
  double const sizeThresholdNominator{static_cast<double>(imageMat.cols)};
  double constexpr SIZE_THRESHOLD_DENOMINATOR{200.0};
  double const sizeThreshold{sizeThresholdNominator /
                             SIZE_THRESHOLD_DENOMINATOR};
  std::vector<hpm::KeyPoint> bigEllipses{
      getBigEllipses(edCircles, sizeThreshold)};

  if (showIntermediateImages) {
    cv::Mat cpy = imageMat.clone();
    drawKeyPoints(cpy, bigEllipses, cv::Scalar(255, 255, 0));
    showImage(cpy, "big-ellipses.png");
  }

  std::vector<hpm::KeyPoint> almostRoundEllipses;
  for (auto const &e : bigEllipses) {
    PixelPosition const center{static_cast<double>(imageMat.cols) / 2.0,
                               static_cast<double>(imageMat.rows) / 2.0};
    PixelPosition const distCoord = e.m_center - center;
    double const dist =
        sqrt(distCoord.x * distCoord.x + distCoord.y * distCoord.y);
    double const maxDist{0.5 * static_cast<double>(imageMat.rows)};
    if (dist < maxDist and e.m_major == e.m_minor) {
      // a circle near middle of image
      almostRoundEllipses.emplace_back(e);
    } else if (e.m_minor * 1.2 > e.m_major and e.m_major != e.m_minor) {
      almostRoundEllipses.emplace_back(e);
    }
  }
  if (showIntermediateImages) {
    cv::Mat cpy = imageMat.clone();
    drawKeyPoints(cpy, almostRoundEllipses, cv::Scalar(255, 255, 0));
    showImage(cpy, "almost-round-ellipses.png");
  }

  if (almostRoundEllipses.empty()) {
    return {};
  }

  // At this point, the legit markers should make up the majority
  // of the big circles in the vector, and their size should be similar
  // So we should have 6 or more markers of the right size
  // and several groups of less than 6 markers that are of the wrong size
  std::sort(almostRoundEllipses.begin(), almostRoundEllipses.end(),
            [&](auto const &lhv, auto const &rhv) {
              return lhv.m_minor < rhv.m_minor;
            });

  std::vector<hpm::KeyPoint> rightSizedEllipses;
  double const medianSize =
      almostRoundEllipses.size() >= 4
          ? almostRoundEllipses[almostRoundEllipses.size() - 4UL].m_minor
          : almostRoundEllipses[0].m_minor;
  for (auto const &e : almostRoundEllipses) {
    if (e.m_minor / medianSize < 1.5 and e.m_minor / medianSize > 0.5) {
      rightSizedEllipses.emplace_back(e);
    }
  }
  if (showIntermediateImages) {
    cv::Mat cpy = imageMat.clone();
    drawKeyPoints(cpy, rightSizedEllipses, cv::Scalar(255, 255, 0));
    showImage(cpy, "right-sized-ellipses.png");
  }

  cv::Mat hsv{};
  cv::cvtColor(imageMat, hsv, cv::COLOR_BGR2HSV);
  cv::Mat hue = getHueChannelCopy(hsv);
  std::vector<HuedKeyPoint> huedEllipses;
  for (auto const &e : rightSizedEllipses) {
    huedEllipses.emplace_back(e, (hue.at<uint8_t>(e.m_center) + 30) % 180);
  }

  std::sort(
      huedEllipses.begin(), huedEllipses.end(),
      [&](auto const &lhv, auto const &rhv) { return lhv.hue < rhv.hue; });

  double const min = static_cast<double>(huedEllipses[0].hue);
  double const max = static_cast<double>(huedEllipses.back().hue);
  double redMid = min;
  double greenMid = huedEllipses[huedEllipses.size() / 2].hue;
  double blueMid = max;

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

static auto angularRange(double f, double semiMinor, double c, double centerRay)
    -> std::pair<double, double> {
  double const semiMajor = semiMinor * sqrt(centerRay * c / (f * f) + 1);
  double const closest = c - semiMajor;
  double const farthest = c + semiMajor;
  double const smallestAng = atan(closest / f);
  double const largestAng = atan(farthest / f);
  return {smallestAng, largestAng};
}

auto ellipseToPosition(hpm::KeyPoint const &ellipse, double focalLength,
                       hpm::PixelPosition const &imageCenter,
                       double markerDiameter) -> hpm::CameraFramedPosition {
  // The ED ellipse detector is good at determining center and minor axes
  // of an ellipse, but very bad at determining the major axis and the rotation.
  // That made this function a bit hard to write.
  double const markerR = markerDiameter / 2;
  double const f = focalLength;
  double const semiMinor = ellipse.m_minor / 2;

  // Luckily, the z position of the marker is determined by the
  // minor axis alone, no need for the major axis or rotation.
  double const z = hpm::zFromSemiMinor(markerR, f, semiMinor);

  // The center of the ellipse is not a projection of the center of the marker.
  // Rather, the center of the marker projects into a point slightly closer
  // to the center of the image, like this
  PixelPosition const imageCenterToEllipseCenter =
      ellipse.m_center - imageCenter;
  double const c = cv::norm(imageCenterToEllipseCenter);
  double const centerRay = centerRayFromZ(c, markerR, z);

  // The center ray and the ellipse center give us the scaling
  // factor between minor and major axis, which lets
  // us compute the angular width and angular position
  // of the cone that gets projected through the pinhole
  auto const [smallestAng, largestAng] =
      angularRange(f, semiMinor, c, centerRay);

  // The angle between the center ray and the image axis
  double const alpha = std::midpoint(largestAng, smallestAng);
  // facing disc's angular radius seen from the pinhole,
  // or "half the cone's inner angle" if you will
  double const theta = std::midpoint(largestAng, -smallestAng);

  // We know that
  //   theta = asin(r/d),
  // where r is markerR,
  // and d is the marker's total distance from the pinhole
  double const d = markerR / sin(theta);

  // Extracting the xy-distance using the angle between the center ray
  // and the image axis
  double const dxy = sin(alpha) * d;

  // Since ed isn't good at finding m_rot, let's calculate the rotation
  // based on the center point, which is more accurately detected by ed.
  double const rot =
      atan2(imageCenterToEllipseCenter.y, imageCenterToEllipseCenter.x);

  return {dxy * cos(rot), dxy * sin(rot), z};
}
