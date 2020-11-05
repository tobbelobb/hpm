#include <string>

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <hpm/hpm.h++>

enum RGB { RED = 2, BLUE = 1, GREEN = 0 };

void drawMarkers(cv::InputArray image,
                 std::vector<cv::KeyPoint> const &keyPoints,
                 cv::InputOutputArray result) {
  const auto BLACK{cv::Scalar(0)};
  cv::drawKeypoints(image, keyPoints, result, BLACK,
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::drawKeypoints(image, keyPoints, result, BLACK,
                    cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
}

void drawMarkers(cv::InputOutputArray image,
                 std::vector<cv::KeyPoint> const &markers) {
  drawMarkers(image, markers, image);
}

void drawCircle(cv::InputOutputArray image, cv::KeyPoint const &marker) {
  const cv::Scalar RED{0, 0, 255};
  cv::circle(image, marker.pt, static_cast<int>(marker.size / 2), RED);
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

auto getSingleColor(cv::InputArray image, enum RGB color) -> cv::Mat {
  cv::Mat singleColorImage{};
  cv::extractChannel(image, singleColorImage, color);
  return singleColorImage;
}

auto getRed(cv::InputArray image) -> cv::Mat {
  return getSingleColor(image, RGB::RED);
}

auto getGreen(cv::InputArray image) -> cv::Mat {
  return getSingleColor(image, RGB::GREEN);
}

auto getBlue(cv::InputArray image) -> cv::Mat {
  return getSingleColor(image, RGB::BLUE);
}

auto getRedInverted(cv::InputArray image) -> cv::Mat {
  cv::Mat redInverted;
  cv::bitwise_not(getRed(image), redInverted);
  return redInverted;
}

auto imageWithMarkers(cv::InputArray image,
                      std::vector<cv::KeyPoint> const &markers) -> cv::Mat {
  cv::Mat result{};
  drawMarkers(image, markers, result);
  return result;
}

auto getBlobDetector() {
  cv::SimpleBlobDetector::Params params = []() {
    cv::SimpleBlobDetector::Params params_;
    params_.thresholdStep = 10.0;       // NOLINT
    params_.minThreshold = 20.0;        // NOLINT
    params_.maxThreshold = 2000.0;      // NOLINT
    params_.minRepeatability = 2;       // NOLINT
    params_.minDistBetweenBlobs = 10.0; // NOLINT
    params_.filterByColor = true;       // NOLINT
    params_.blobColor = 0;              // NOLINT
    params_.filterByArea = true;        // NOLINT
    params_.minArea = 250.0;            // NOLINT
    params_.maxArea = 500000.0;         // NOLINT
    params_.filterByCircularity = true; // NOLINT
    params_.minCircularity = 0.8;       // NOLINT
    params_.maxCircularity = 3.4e38;    // NOLINT
    params_.filterByInertia = true;     // NOLINT
    params_.minInertiaRatio = 0.1;      // NOLINT
    params_.maxInertiaRatio = 3.4e38;   // NOLINT
    params_.filterByConvexity = true;   // NOLINT
    params_.minConvexity = 0.95;        // NOLINT
    params_.maxConvexity = 3.4e38;      // NOLINT
    return params_;
  }();

  cv::Ptr<cv::Feature2D> simpleBlobDetector =
      cv::SimpleBlobDetector::create(params);

  return simpleBlobDetector;
}

auto detect(cv::InputArray image, cv::Ptr<cv::Feature2D> const &detector) {
  std::vector<cv::KeyPoint> keyPoints{};
  detector->detect(image, keyPoints);
  return keyPoints;
}

// Algorithm: Blob Detector
// See:
// https://docs.opencv.org/4.4.0/d0/d7a/classcv_1_1SimpleBlobDetector.html#details
// and also
// https://www.learnopencv.com/blob-detection-using-opencv-python-c/
auto blobDetect(cv::InputArray image) {
  auto const detector = getBlobDetector();
  // Combine channels in a way that turns markers into dark regions
  constexpr double THIRD{0.33333};
  cv::Mat combined = getGreen(image) * THIRD + getBlue(image) * THIRD +
                     getRedInverted(image) * THIRD;

  return detect(combined, detector);
}

auto detectMarkers(cv::InputArray undistortedImage, bool showIntermediateImages)
    -> std::vector<cv::KeyPoint> {
  std::vector<cv::KeyPoint> markers{blobDetect(undistortedImage)};
  if (showIntermediateImages) {
    showImage(imageWithMarkers(undistortedImage, markers),
              "simpleBlobDetector.png");
  }
  return markers;
}

auto toCameraPosition(cv::KeyPoint const &keyPoint, double focalLength,
                      double markerDiameter) -> Position {
  return {0, 0, markerDiameter * focalLength / keyPoint.size};
}
