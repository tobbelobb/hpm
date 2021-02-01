#include <hpm/ellipse-detector.h++>
#include <hpm/marks.h++>
#include <hpm/util.h++>

#include <hpm/ed/EDLib.h++>

#include <algorithm>
#include <cmath>

using namespace hpm;

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

static auto bgr2skewedHue(cv::Point3_<uint8_t> bgr) -> double {
  auto smallHue = [](std::array<double, 3> const &rgb) -> double {
    auto const [min, max] = std::minmax_element(std::begin(rgb), std::end(rgb));
    if (max == std::begin(rgb)) {
      return (rgb[1] - rgb[2]) / (*max - *min);
    }
    if (max == std::next(std::begin(rgb))) {
      return 2.0 + (rgb[2] - rgb[0]) / (*max - *min);
    }
    return 4.0 + (rgb[0] - rgb[1]) / (*max - *min);
  };
  std::array<double, 3> const rgb = {static_cast<double>(bgr.z) / 255.0,
                                     static_cast<double>(bgr.y) / 255.0,
                                     static_cast<double>(bgr.x) / 255.0};
  constexpr double MAX_DEG{360.0};
  double hue = smallHue(rgb);
  hue *= 60.0;
  if (hue < 0.0) {
    hue += MAX_DEG;
  }
  assert(hue < MAX_DEG);
  assert(hue >= 0.0);
  double const ret = std::fmod(hue + 60.0, MAX_DEG);
  // skew hue so all red colors end up closer to 0 than to MAX_DEG.
  return ret;
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
  const auto AQUA{cv::Scalar(255, 255, 0)};
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
    for (auto const &ellipse : bigEllipses) {
      draw(cpy, ellipse, AQUA);
    }
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
    for (auto const &ellipse : almostRoundEllipses) {
      draw(cpy, ellipse, AQUA);
    }
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
  for (auto const &ellipse : almostRoundEllipses) {
    if (ellipse.m_minor / medianSize < 1.5 and
        ellipse.m_minor / medianSize > 0.5) {
      rightSizedEllipses.emplace_back(ellipse);
    }
  }
  if (showIntermediateImages) {
    cv::Mat cpy = imageMat.clone();
    for (auto const &ellipse : rightSizedEllipses) {
      draw(cpy, ellipse, AQUA);
    }
    showImage(cpy, "right-sized-ellipses.png");
  }

  std::vector<hpm::Mark> marks;
  for (auto const &e : rightSizedEllipses) {
    // The ellipse center might be outside of picture frame
    int const row{
        std::clamp(static_cast<int>(e.m_center.y), 0, imageMat.rows - 1)};
    int const col{
        std::clamp(static_cast<int>(e.m_center.x), 0, imageMat.cols - 1)};
    auto bgr = imageMat.at<cv::Point3_<uint8_t>>(row, col);
    double const skewedHue = bgr2skewedHue(bgr);

    marks.emplace_back(e, skewedHue);
  }

  std::sort(marks.begin(), marks.end(), [&](auto const &lhv, auto const &rhv) {
    return lhv.m_hue < rhv.m_hue;
  });

  double const min = marks[0].m_hue;
  double const max = marks.back().m_hue;
  double redMid = min;
  double greenMid = marks[marks.size() / 2].m_hue;
  double blueMid = max;

  Marks result{};
  for (auto const &mark : marks) {
    double colorDistanceRed = std::abs(mark.m_hue - redMid);
    double colorDistanceGreen = std::abs(mark.m_hue - greenMid);
    double colorDistanceBlue = std::abs(mark.m_hue - blueMid);
    if (colorDistanceRed < colorDistanceGreen and
        colorDistanceRed < colorDistanceBlue) {
      result.m_red.emplace_back(mark);
    } else if (colorDistanceGreen < colorDistanceRed and
               colorDistanceGreen < colorDistanceBlue) {
      result.m_green.emplace_back(mark);
    } else {
      result.m_blue.emplace_back(mark);
    }
  }

  return result;
}
