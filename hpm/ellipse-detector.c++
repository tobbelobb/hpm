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
  if (showIntermediateImages) {
    showImage(edColor.getEdgeImage(), "edgeImage.png");
    cv::Mat cpy(imageMat.rows, imageMat.cols, CV_8UC3, WHITE);
    std::random_device
        rd; // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(0, 255); // NOLINT
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

auto ellipseDetect(cv::InputArray image, bool showIntermediateImages)
    -> std::vector<hpm::Ellipse> {
  cv::Mat imageMat{image.getMat()};

  std::vector<hpm::Ellipse> ellipses{
      rawEllipseDetect(image, showIntermediateImages)};

  // Size of a marker must be at least 1/200 of the image width
  double const sizeThresholdNominator{static_cast<double>(imageMat.cols)};
  double constexpr SIZE_THRESHOLD_DENOMINATOR{200.0};
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
    double constexpr MAX_MAJOR_MINOR_RATIO{1.4};
    if (e.m_minor * MAX_MAJOR_MINOR_RATIO > e.m_major) {
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
