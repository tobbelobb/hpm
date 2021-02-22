#include <hpm/ed/EDLib.h++>
#include <hpm/test-util.h++> // getPath

#include <boost/ut.hpp> //import boost.ut;

#include <iostream>

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;
  cv::Mat billiardColor{cv::imread(getPath("billiard.jpg"), cv::IMREAD_COLOR)};
  cv::Mat billiardGrey{
      cv::imread(getPath("billiard.jpg"), cv::IMREAD_GRAYSCALE)};

  "Billiard Segments from grey image"_test = [billiardGrey] {
    ED ed{billiardGrey,
          {.op = GradientOperator::SOBEL,
           .gradThresh = 36,
           .anchorThresh = 8,
           .scanInterval = 1,
           .blurSize = 1.0,
           .sumFlag = true}};
    expect(ed.getSegmentNo() == 415_i);
  };

  "Billiard Lines from grey image"_test = [billiardGrey] {
    EDLines edLines{billiardGrey};
    expect(edLines.getLinesNo() == 518_i);
  };

  "Billiard Lines from segments"_test = [billiardGrey] {
    ED ed{billiardGrey,
          {.op = GradientOperator::SOBEL, .gradThresh = 36, .anchorThresh = 8}};
    EDLines edLines{ed};
    expect(edLines.getLinesNo() == 518_i);
  };

  "Billiard Parameter free segments from grey image"_test = [billiardGrey] {
    EDPF edpf{billiardGrey};
    expect(edpf.getSegmentNo() == 276_i);
  };

  "Billiard Circles from grey image"_test = [billiardGrey] {
    EDCircles edCirclesFromGreyImage{billiardGrey};
    expect(edCirclesFromGreyImage.getCirclesNo() == 16_i);
  };

  "Billiard Circles from parameter free segments"_test = [billiardGrey] {
    EDPF edpf{billiardGrey};
    EDCircles edCirclesFromEdpf{edpf};
    expect(edCirclesFromEdpf.getCirclesNo() == 16_i);
  };

  "Billiard Segments from color image"_test = [billiardColor] {
    EDColor edColor{billiardColor,
                    {.gradThresh = 36,
                     .anchorThresh = 4,
                     .blurSize = 1.5,
                     .filterSegments = true}};
    expect(edColor.getNumberOfSegments() == 212_u);
  };

  "Billiard Lines from segments from color image"_test = [billiardColor] {
    EDColor edColor{billiardColor,
                    {.gradThresh = 36,
                     .anchorThresh = 4,
                     .blurSize = 1.5,
                     .filterSegments = true}};
    EDLines colorLines{edColor};
    expect(colorLines.getLinesNo() == 585_i);
  };

  "Billiard Circles from segments from color image"_test = [billiardColor] {
    EDColor testEDColor{billiardColor,
                        {.gradThresh = 36,
                         .anchorThresh = 4,
                         .blurSize = 1.5,
                         .filterSegments = true}};
    EDCircles colorCircle{testEDColor};
    expect(colorCircle.getCirclesNo() == 46_i);
  };

  "Billiard color no validate segments"_test = [billiardColor] {
    EDColor testEDColor{billiardColor,
                        {.gradThresh = 36,
                         .anchorThresh = 4,
                         .blurSize = 1.5,
                         .filterSegments = false}};
    expect(testEDColor.getNumberOfSegments() == 230_u);

    EDLines colorLine = EDLines(testEDColor);
    expect(colorLine.getLinesNo() == 571_i);

    EDCircles colorCircle{testEDColor};
    expect(colorCircle.getCirclesNo() == 45_i);
  };

  "Circles from red filled circle"_test = [] {
    auto constexpr rows{600};
    auto constexpr cols{600};
    cv::Scalar const WHITE{255, 255, 255};
    cv::Point3_<uint8_t> const RED{0, 0, 255};
    cv::Point2d const imageCenter{static_cast<double>(cols) / 2.0,
                                  static_cast<double>(rows) / 2.0};
    for (double radius{5.0};
         radius < static_cast<double>(std::min(rows, cols)) / 2.0;
         radius += 10.0) {
      cv::Mat image{rows, cols, CV_8UC3, WHITE};

      image.forEach<cv::Point3_<uint8_t>>(
          [&RED, &radius, &imageCenter](auto &point, int const positions[]) {
            // Position of pixel is at the middle of the pixel
            cv::Point2d const pos{static_cast<double>(positions[0]) + 0.5,
                                  static_cast<double>(positions[1]) + 0.5};
            double const dist = cv::norm(pos - imageCenter);
            if (dist < radius) {
              point = RED;
            }
          });

      EDColor testEDColor{image,
                          {.gradThresh = 36,
                           .anchorThresh = 4,
                           .blurSize = 1.5,
                           .filterSegments = true}};
      EDCircles colorCircle{testEDColor};
      expect((colorCircle.getCirclesNo() == 1_i) >> fatal);
      auto const circle{colorCircle.getCirclesRef()[0]};
      auto constexpr EPS_CENTER{0.04_d};
      auto constexpr EPS_RADIUS{0.20_d};
      // To encapsulate all the colored pixels, we must go half a pixel further
      // out compared to the condition (dist < radius) above
      auto const encapsulating_radius{radius + 0.5};
      expect(std::abs(imageCenter.x - circle.center.x) < EPS_CENTER);
      expect(std::abs(imageCenter.y - circle.center.y) < EPS_CENTER);
      expect(std::abs(encapsulating_radius - circle.r) < EPS_RADIUS);
    }
  };

  return 0;
}
