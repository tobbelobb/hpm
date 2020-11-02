#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <hpm/hpm.h++>

void showImage(cv::InputArray image, std::string const &name) {
  cv::namedWindow(name, cv::WINDOW_NORMAL);
  constexpr auto SHOW_PIXELS_X{300};
  constexpr auto SHOW_PIXELS_Y{300};
  cv::resizeWindow(name, SHOW_PIXELS_X, SHOW_PIXELS_Y);
  cv::imshow(name, image);
  if (cv::waitKey(0) == 's') {
    cv::imwrite(name, image);
  }
}

std::vector<Marker> detectMarkers(CamParams const &camParams,
                                  cv::InputArray const undistortedImage,
                                  bool showIntermediateImages) {
  if (showIntermediateImages) {
    showImage(undistortedImage, "intermediate0.png");
  }
  return {};
}

void draw(/* inout */ cv::Mat image, Marker const &marker) {
  cv::circle(image, {static_cast<int>(marker.x), static_cast<int>(marker.y)},
             static_cast<int>(marker.r), cv::Scalar{255, 0, 0}, -1, 8, 0);
}
