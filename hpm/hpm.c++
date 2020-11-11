#include <string>

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <hpm/hpm.h++>

enum RGB { BLUE = 0, GREEN = 1, RED = 2 };

static void drawMarkers(cv::InputArray image,
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

static auto getSingleColor(cv::InputArray image, enum RGB color) -> cv::Mat {
  cv::Mat singleColorImage{};
  cv::extractChannel(image, singleColorImage, color);
  return singleColorImage;
}

static auto getRed(cv::InputArray image) -> cv::Mat {
  return getSingleColor(image, RGB::RED);
}

static auto getGreen(cv::InputArray image) -> cv::Mat {
  return getSingleColor(image, RGB::GREEN);
}

static auto getBlue(cv::InputArray image) -> cv::Mat {
  return getSingleColor(image, RGB::BLUE);
}

static auto invert(cv::InputArray image) -> cv::Mat {
  cv::Mat inverted;
  cv::bitwise_not(image, inverted);
  return inverted;
}

// auto invert1of3(cv::InputArray image, enum RGB color) -> cv::Mat {
//  cv::Mat imageClone = image.clone();
//  for (auto &pixel : imageClone) {
//    pixel[color] = cv::bitwise_not(pixel);
//  }
//  return imageClone;
//}

static auto imageWithMarkers(cv::InputArray image,
                             std::vector<cv::KeyPoint> const &markers)
    -> cv::Mat {
  cv::Mat result{};
  drawMarkers(image, markers, result);
  return result;
}

static auto getBlobDetector() {
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

static auto detect(cv::InputArray image,
                   cv::Ptr<cv::Feature2D> const &detector) {
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
  cv::Mat antiRed = getGreen(image) * THIRD + getBlue(image) * THIRD +
                    invert(getRed(image)) * THIRD;
  cv::Mat antiBlue = getGreen(image) * THIRD + invert(getBlue(image)) * THIRD +
                     getRed(image) * THIRD;
  // Green seems harder to find on OpenScad images...
  cv::Mat antiGreen = invert(getGreen(image)) * 0.25 +
                      getBlue(image) * 0.75 / 2 + getRed(image) * 0.75 / 2;

  // The three detect lines take up ~90% of execution time
  // auto const t0{cv::getTickCount()};
  auto redMarkers{detect(antiRed, detector)};
  auto greenMarkers{detect(antiGreen, detector)};
  auto blueMarkers{detect(antiBlue, detector)};
  // auto const t1{cv::getTickCount()};
  // std::cout << "Time to detect: " << (t1 - t0) / cv::getTickFrequency()
  //          << " seconds" << std::endl;

  std::vector<cv::KeyPoint> keyPoints{};
  keyPoints.reserve(redMarkers.size() + blueMarkers.size() +
                    greenMarkers.size());
  keyPoints.insert(keyPoints.end(), blueMarkers.begin(), blueMarkers.end());
  keyPoints.insert(keyPoints.end(), greenMarkers.begin(), greenMarkers.end());
  keyPoints.insert(keyPoints.end(), redMarkers.begin(), redMarkers.end());

  return keyPoints;
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
                      cv::Point2f const &imageCenter, double markerDiameter)
    -> Position {
  auto const fromCenter{keyPoint.pt - imageCenter};
  return {fromCenter.x * markerDiameter / keyPoint.size,
          -fromCenter.y * markerDiameter / keyPoint.size,
          focalLength * markerDiameter / keyPoint.size};
}
