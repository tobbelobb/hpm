#include <hpm/ellipse-detector.h++>
#include <hpm/marks.h++>
#include <hpm/util.h++>

#include <hpm/ed/EDLib.h++>

#include <algorithm>
#include <cmath>
#include <random>

using namespace hpm;

static auto getBigEllipses(std::vector<hpm::Ellipse> const &ellipses,
                           double sizeThreshold) -> std::vector<hpm::Ellipse> {
  std::vector<hpm::Ellipse> bigEllipses{};
  for (auto const &ellipse : ellipses) {
    if (ellipse.m_minor > sizeThreshold) {
      bigEllipses.emplace_back(ellipse);
    }
  }
  return bigEllipses;
}

auto rawEllipseDetect(cv::InputArray image, bool showIntermediateImages)
    -> std::vector<hpm::Ellipse> {
  cv::Mat imageMat{image.getMat()};
  EDColor const edColor{
      imageMat,
      {.gradThresh = 20, // lower gradThresh finds more ellipses, both true and
                         // false positives
       .anchorThresh = 4,
       .blurSize = 1.5,
       .filterSegments = true}};
  const auto AQUA{cv::Scalar(255, 255, 0)};
  if (showIntermediateImages) {
    showImage(edColor.getEdgeImage(), "edgeImage.png");
    cv::Mat cpy(imageMat.rows, imageMat.cols, CV_8UC3,
                cv::Scalar(255, 255, 255));
    std::random_device
        rd; // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(0, 255);
    for (auto const &segment : edColor.getSegments()) {
      uint8_t first{static_cast<uint8_t>(distrib(gen))};
      uint8_t second{static_cast<uint8_t>(distrib(gen))};
      uint8_t third{static_cast<uint8_t>(distrib(gen))};
      for (auto const &point : segment) {
        cpy.at<cv::Point3_<uint8_t>>(point) =
            cv::Point3_<uint8_t>(first, second, third);
      }
    }
    showImage(cpy, "segmentImage.png");
  }

  EDCircles const edCircles{edColor};
  if (showIntermediateImages) {
    showImage(edCircles.drawResult(imageMat, ImageStyle::BOTH),
              "edCircles.png");
  }

  std::vector<hpm::Ellipse> ellipses{};
  for (auto const &circle : edCircles.getCirclesRef()) {
    ellipses.emplace_back(circle);
  }
  for (auto const &ellipse : edCircles.getEllipsesRef()) {
    ellipses.emplace_back(ellipse);
  }
  return ellipses;
}

auto ellipseDetect(cv::InputArray image, bool showIntermediateImages,
                   PixelPosition const &expectedTopLeftestCenter)
    -> std::vector<hpm::Ellipse> {
  cv::Mat imageMat{image.getMat()};
  const auto AQUA{cv::Scalar(255, 255, 0)};

  std::vector<hpm::Ellipse> ellipses{
      rawEllipseDetect(image, showIntermediateImages)};

  auto closerToTopLeft = [&expectedTopLeftestCenter](hpm::Ellipse const &lhs,
                                                     hpm::Ellipse const &rhs) {
    return cv::norm(lhs.m_center - expectedTopLeftestCenter) <
           cv::norm(rhs.m_center - expectedTopLeftestCenter);
  };
  if (not ellipses.empty() and
      expectedTopLeftestCenter != PixelPosition(0.0, 0.0)) {
    Ellipse const topLeftest{*std::min_element(
        std::begin(ellipses), std::end(ellipses), closerToTopLeft)};
    PixelPosition const imageFrameShift{topLeftest.m_center -
                                        expectedTopLeftestCenter};
    if (cv::norm(imageFrameShift) < 10.0) {
      for (auto &e : ellipses) {
        e.m_center -= imageFrameShift;
      }
    }
  }

  // Size of a marker must be at least 1/250 of the image width
  double const sizeThresholdNominator{static_cast<double>(imageMat.cols)};
  double constexpr SIZE_THRESHOLD_DENOMINATOR{250.0};
  double const sizeThreshold{sizeThresholdNominator /
                             SIZE_THRESHOLD_DENOMINATOR};
  std::vector<hpm::Ellipse> const bigEllipses{
      getBigEllipses(ellipses, sizeThreshold)};

  if (showIntermediateImages) {
    cv::Mat cpy = imageMat.clone();
    for (auto const &ellipse : bigEllipses) {
      draw(cpy, ellipse, AQUA);
    }
    showImage(cpy, "big-ellipses.png");
  }

  std::vector<hpm::Ellipse> almostRoundEllipses;
  for (auto const &e : bigEllipses) {
    if (e.m_minor * 1.4 > e.m_major) {
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

  return almostRoundEllipses;
}
