#include <opencv2/imgproc.hpp>

#include <hpm/hpm.h++>

std::vector<Marker> detectMarkers(CamParams const &, cv::Mat const image) {
  return {};
}

void draw(/* inout */ cv::Mat image, Marker const &marker) {
  cv::circle(image, {static_cast<int>(marker.x), static_cast<int>(marker.y)},
             static_cast<int>(marker.r), cv::Scalar{255, 0, 0}, -1, 8, 0);
}
