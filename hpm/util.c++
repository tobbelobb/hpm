
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

void drawKeyPoints(cv::InputOutputArray image,
                   std::vector<hpm::KeyPoint> const &keyPoints,
                   cv::Scalar const &color) {
  for (auto const &keyPoint : keyPoints) {
    cv::circle(image, keyPoint.center, static_cast<int>(keyPoint.size / 2.0),
               color, 3);
    cv::circle(image, keyPoint.center, 2, color, 3);
  }
}

void drawDetectionResult(cv::InputOutputArray image,
                         DetectionResult const &markers) {
  const auto AQUA{cv::Scalar(255, 255, 0)};
  const auto FUCHSIA{cv::Scalar(255, 0, 255)};
  const auto YELLOW{cv::Scalar(0, 255, 255)};
  drawKeyPoints(image, markers.red, AQUA);
  drawKeyPoints(image, markers.green, FUCHSIA);
  drawKeyPoints(image, markers.blue, YELLOW);
}

auto imageWithDetectionResult(cv::InputArray image,
                              DetectionResult const &detectionResult)
    -> cv::Mat {
  cv::Mat imageCopy{};
  image.copyTo(imageCopy);
  drawDetectionResult(imageCopy, detectionResult);
  return imageCopy;
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

auto ScalarBGR2HSV(cv::Scalar const &bgr) -> cv::Scalar {
  cv::Mat const bgrMat{1, 1, CV_8UC3, bgr};
  cv::Mat hsv{1, 1, CV_8UC3, bgr};
  cv::cvtColor(bgrMat, hsv, cv::COLOR_BGR2HSV);
  return cv::Scalar{static_cast<double>(hsv.data[0]),
                    static_cast<double>(hsv.data[1]),
                    static_cast<double>(hsv.data[2]), 1.0};
}
