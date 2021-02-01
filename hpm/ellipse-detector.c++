#include <hpm/ellipse-detector.h++>
#include <hpm/marks.h++>
#include <hpm/util.h++>

#include <hpm/ed/EDLib.h++>

#include <algorithm>

using namespace hpm;

struct HuedEllipse {
  hpm::Ellipse ellipse;
  uint8_t hue;
  HuedEllipse(hpm::Ellipse const e, uint8_t hue_in) : ellipse(e), hue(hue_in) {}
};

static auto getBigEllipses(EDCircles const &edCircles, double sizeThreshold)
    -> std::vector<hpm::Ellipse> {
  std::vector<hpm::Ellipse> bigEllipses{};
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
    -> hpm::Marks {
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
  std::vector<hpm::Ellipse> bigEllipses{
      getBigEllipses(edCircles, sizeThreshold)};

  if (showIntermediateImages) {
    cv::Mat cpy = imageMat.clone();
    drawMarks(cpy, bigEllipses, cv::Scalar(255, 255, 0));
    showImage(cpy, "big-ellipses.png");
  }

  std::vector<hpm::Ellipse> almostRoundEllipses;
  for (auto const &e : bigEllipses) {
    PixelPosition const center{static_cast<double>(imageMat.cols) / 2.0,
                               static_cast<double>(imageMat.rows) / 2.0};
    PixelPosition const distCoord = e.m_center - center;
    double const dist =
        sqrt(distCoord.x * distCoord.x + distCoord.y * distCoord.y);
    double const maxDist{0.6 * static_cast<double>(imageMat.rows)};
    if (dist < maxDist and e.m_major == e.m_minor) {
      // a circle near middle of image
      almostRoundEllipses.emplace_back(e);
    } else if (e.m_minor * 1.2 > e.m_major and e.m_major != e.m_minor) {
      almostRoundEllipses.emplace_back(e);
    }
  }
  if (showIntermediateImages) {
    cv::Mat cpy = imageMat.clone();
    drawMarks(cpy, almostRoundEllipses, cv::Scalar(255, 255, 0));
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

  std::vector<hpm::Ellipse> rightSizedEllipses;
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
    drawMarks(cpy, rightSizedEllipses, cv::Scalar(255, 255, 0));
    showImage(cpy, "right-sized-ellipses.png");
  }

  cv::Mat hsv{};
  cv::cvtColor(imageMat, hsv, cv::COLOR_BGR2HSV);
  // TODO: We should not compute hue of every pixel when we only need
  //       the hue of ~10 pixels.
  cv::Mat hue = getHueChannelCopy(hsv);
  std::vector<HuedEllipse> huedEllipses;
  for (auto const &e : rightSizedEllipses) {
    // The ellipse center might be outside of picture frame
    int const row{
        std::clamp(static_cast<int>(e.m_center.y), 0, imageMat.rows - 1)};
    int const col{
        std::clamp(static_cast<int>(e.m_center.x), 0, imageMat.cols - 1)};
    huedEllipses.emplace_back(e, (hue.at<uint8_t>(row, col) + 30) % 180);
  }

  std::sort(
      huedEllipses.begin(), huedEllipses.end(),
      [&](auto const &lhv, auto const &rhv) { return lhv.hue < rhv.hue; });

  double const min = static_cast<double>(huedEllipses[0].hue);
  double const max = static_cast<double>(huedEllipses.back().hue);
  double redMid = min;
  double greenMid = huedEllipses[huedEllipses.size() / 2].hue;
  double blueMid = max;

  Marks result{};
  for (auto const &he : huedEllipses) {
    double colorDistanceRed = std::abs(static_cast<double>(he.hue) - redMid);
    double colorDistanceGreen =
        std::abs(static_cast<double>(he.hue) - greenMid);
    double colorDistanceBlue = std::abs(static_cast<double>(he.hue) - blueMid);
    if (colorDistanceRed < colorDistanceGreen and
        colorDistanceRed < colorDistanceBlue) {
      result.red.emplace_back(he.ellipse);
    } else if (colorDistanceGreen < colorDistanceRed and
               colorDistanceGreen < colorDistanceBlue) {
      result.green.emplace_back(he.ellipse);
    } else {
      result.blue.emplace_back(he.ellipse);
    }
  }

  return result;
}
