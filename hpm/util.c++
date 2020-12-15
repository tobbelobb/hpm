
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#pragma GCC diagnostic pop

#include <hpm/util.h++>

using namespace hpm;

void drawKeyPoints(cv::InputArray image,
                   std::vector<hpm::KeyPoint> const &keyPoints,
                   cv::InputOutputArray result) {
  const auto BLACK{cv::Scalar(0)};
  std::vector<cv::KeyPoint> cvKeyPoints{};
  std::transform(keyPoints.begin(), keyPoints.end(),
                 std::back_inserter(cvKeyPoints),
                 [](hpm::KeyPoint const &keyPoint) { return keyPoint.toCv(); });
  cv::drawKeypoints(image, cvKeyPoints, result, BLACK,
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::drawKeypoints(image, cvKeyPoints, result, BLACK,
                    cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
}

auto imageWithKeyPoints(cv::InputArray image, DetectionResult const &markers)
    -> cv::Mat {
  cv::Mat result{};
  drawKeyPoints(image, markers.red, result);
  drawKeyPoints(result, markers.green, result);
  drawKeyPoints(result, markers.blue, result);
  return result;
}

void showImage(cv::InputArray image, std::string const &name) {
  cv::namedWindow(name, cv::WINDOW_NORMAL);
  constexpr auto SHOW_PIXELS_X{1500};
  constexpr auto SHOW_PIXELS_Y{1500};
  cv::resizeWindow(name, SHOW_PIXELS_X, SHOW_PIXELS_Y);
  cv::imshow(name, image);
  if (cv::waitKey(0) == 's') {
    cv::imwrite(name, image);
  }
}
