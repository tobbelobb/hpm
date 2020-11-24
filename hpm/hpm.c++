#include <cmath> // atan
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#pragma GCC diagnostic pop

#include <hpm/hpm.h++>

static void drawKeyPoints(cv::InputArray image,
                          std::vector<cv::KeyPoint> const &keyPoints,
                          cv::InputOutputArray result) {
  const auto BLACK{cv::Scalar(0)};
  cv::drawKeypoints(image, keyPoints, result, BLACK,
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::drawKeypoints(image, keyPoints, result, BLACK,
                    cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
}

static auto imageWithKeyPoints(cv::InputArray image,
                               std::vector<cv::KeyPoint> const &markers)
    -> cv::Mat {
  cv::Mat result{};
  drawKeyPoints(image, markers, result);
  return result;
}

static void showImage(cv::InputArray image, std::string const &name) {
  cv::namedWindow(name, cv::WINDOW_NORMAL);
  constexpr auto SHOW_PIXELS_X{1500};
  constexpr auto SHOW_PIXELS_Y{1500};
  cv::resizeWindow(name, SHOW_PIXELS_X, SHOW_PIXELS_Y);
  cv::imshow(name, image);
  if (cv::waitKey(0) == 's') {
    cv::imwrite(name, image);
  }
}

auto find(cv::InputArray undistortedImage, double const knownMarkerDiameter,
          double const focalLength, cv::Point2f const &imageCenter,
          bool showIntermediateImages, bool showResultImage)
    -> std::vector<CameraFramedPosition> {
  if (undistortedImage.empty()) {
    return {};
  }

  auto const blobs{blobDetect(undistortedImage)};

  if (showIntermediateImages or showResultImage) {
    showImage(imageWithKeyPoints(undistortedImage, blobs),
              "markersDetected.png");
  }

  std::vector<CameraFramedPosition> positions{};
  positions.reserve(blobs.size());
  for (auto const &blob : blobs) {
    positions.emplace_back(blobToPosition(blob, focalLength, imageCenter,
                                          knownMarkerDiameter));
  }

  return positions;
}
