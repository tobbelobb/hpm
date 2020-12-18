#include <cmath>
#include <numeric>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#pragma GCC diagnostic pop

#include <hpm/blob-detector.h++>
#include <hpm/detection-result.h++>
#include <hpm/util.h++>

using namespace hpm;

static auto getSingleChannelCopy(cv::InputArray image, int channel) -> cv::Mat {
  cv::Mat singleColorImage{};
  cv::extractChannel(image, singleColorImage, channel);
  return singleColorImage;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static auto getRedCopy(cv::InputArray image) -> cv::Mat {
  return getSingleChannelCopy(image, 2);
}

static auto getGreenCopy(cv::InputArray image) -> cv::Mat {
  return getSingleChannelCopy(image, 1);
}

static auto getBlueCopy(cv::InputArray image) -> cv::Mat {
  return getSingleChannelCopy(image, 0);
}

static auto getValueChannelCopy(cv::InputArray image) -> cv::Mat {
  return getSingleChannelCopy(image, 2);
}

static auto getSaturationChannelCopy(cv::InputArray image) -> cv::Mat {
  return getSingleChannelCopy(image, 1);
}

static auto getHueChannelCopy(cv::InputArray image) -> cv::Mat {
  return getSingleChannelCopy(image, 0);
}

static auto invert(cv::InputArray image) -> cv::Mat {
  cv::Mat inverted;
  cv::bitwise_not(image, inverted);
  return inverted;
}
#pragma GCC diagnostic pop

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
    params_.minArea = 1700.0F;          // NOLINT
    params_.maxArea = 500000.0;         // NOLINT
    params_.filterByCircularity = true; // NOLINT
    params_.minCircularity = 0.70F;     // NOLINT
    params_.maxCircularity = 3.4e38F;   // NOLINT
    params_.filterByInertia = true;     // NOLINT
    params_.minInertiaRatio = 0.1F;     // NOLINT
    params_.maxInertiaRatio = 3.4e38F;  // NOLINT
    params_.filterByConvexity = true;   // NOLINT
    params_.minConvexity = 0.95F;       // NOLINT
    params_.maxConvexity = 3.4e38F;     // NOLINT
    return params_;
  }();
  cv::Ptr<cv::Feature2D> simpleBlobDetector =
      cv::SimpleBlobDetector::create(params);

  return simpleBlobDetector;
}

static auto detect(cv::InputArray image, cv::Ptr<cv::Feature2D> const &detector)
    -> std::vector<hpm::KeyPoint> {
  std::vector<cv::KeyPoint> blobs_{};
  detector->detect(image, blobs_);
  std::vector<hpm::KeyPoint> blobs{};
  std::transform(blobs_.begin(), blobs_.end(), std::back_inserter(blobs),
                 [](cv::KeyPoint const &kp) { return hpm::KeyPoint(kp); });
  return blobs;
}

static auto bellCurve(double const x, double const center,
                      double const standardDeviation) {
  return exp(-0.5 * pow(((x - center) / standardDeviation), 2));
}

// Algorithm: Blob Detector
// See:
// https://docs.opencv.org/4.4.0/d0/d7a/classcv_1_1SimpleBlobDetector.html#details
// and also
// https://www.learnopencv.com/blob-detection-using-opencv-python-c/
auto blobDetect(cv::InputArray image, bool showIntermediateImages)
    -> DetectionResult {
  auto const detector = getBlobDetector();

  cv::Mat imageMat{image.getMat()};

  cv::Mat hsv{};
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

  // Massaging the image to the blobDetector's delight...
  // This is heavily manually done, and very fragile.
  // The whole marker setup must be changed to get rid of this
  // fragile code here.
  auto const BELL_WIDTH{10};
  cv::Mat antiRed = getHueChannelCopy(hsv);
  antiRed.forEach<uint8_t>([](uint8_t &x, const int *position) {
    x = static_cast<uint8_t>(
        255.0 * (1.0 - bellCurve(x, 0, 2) - bellCurve(x, 179, BELL_WIDTH)));
    (void)position;
  });

  cv::Mat antiGreen = getHueChannelCopy(hsv);
  antiGreen.forEach<uint8_t>([](uint8_t &x, const int *position) {
    x = static_cast<uint8_t>(255.0 * (1.0 - bellCurve(x, 60, BELL_WIDTH)));
    (void)position;
  });

  cv::Mat antiBlue = getHueChannelCopy(hsv);
  antiBlue.forEach<uint8_t>([](uint8_t &x, const int *position) {
    x = static_cast<uint8_t>(255.0 * (1.0 - bellCurve(x, 120, BELL_WIDTH)));
    (void)position;
  });

  // Special case for a completely white background
  // White is represented confusingly similar to red in HSV.
  // white is represented like hue=red (that is close to 0 or 179),
  // and saturation~=0
  cv::Mat const saturation = getSaturationChannelCopy(hsv);
  cv::Mat const hue = getHueChannelCopy(hsv);
  for (auto r{0}; r < antiRed.rows; ++r) {
    for (auto c{0}; c < antiRed.cols; ++c) {
      if (saturation.at<uint8_t>(r, c) < 50)
        antiRed.at<uint8_t>(r, c) = 255;
    }
  }

  auto const BLUR_SIZE{7};
  auto const BLUR_DEV{2};
  cv::GaussianBlur(antiRed, antiRed, {BLUR_SIZE, BLUR_SIZE}, BLUR_DEV);
  cv::GaussianBlur(antiGreen, antiGreen, {BLUR_SIZE, BLUR_SIZE}, BLUR_DEV);
  cv::GaussianBlur(antiBlue, antiBlue, {BLUR_SIZE, BLUR_SIZE}, BLUR_DEV);

  if (showIntermediateImages) {
    showImage(getHueChannelCopy(hsv), "hue.png");
    showImage(getSaturationChannelCopy(hsv), "saturation.png");
    showImage(getValueChannelCopy(hsv), "value.png");
    showImage(antiRed, "antiRed.png");
    showImage(antiGreen, "antiGreen.png");
    showImage(antiBlue, "antiBlue.png");
  }

  // With SimpleBlobDetector the three detect lines are very expensive,
  // like ~90% of execution time
  return {detect(antiRed, detector), detect(antiGreen, detector),
          detect(antiBlue, detector)};
}

auto blobDetect(cv::InputArray image) -> DetectionResult {
  return blobDetect(image, false);
}

auto blobToPosition(hpm::KeyPoint const &blob, double const focalLength,
                    PixelPosition const &imageCenter,
                    double const markerDiameter) -> CameraFramedPosition {
  // Step 1: We have an ellipsis within an xy-direced square with a given
  // size
  //         So we know its semi-major axis size and direction,
  //         We also know the position of its center
  PixelPosition const fromCenter = blob.center - imageCenter;
  // This approximation works well close to x and y axis, but
  // if keyPoint lies along y = x, then this approximation isn't good

  // The semi major axis is oriented along this direction
  PixelPosition const dirToOrigin = -fromCenter / cv::norm(fromCenter);

  // These were empirically determined for the SimpleBlobDetector
  // xyOffness:
  // How much does the reported marker size grow or shrink
  // when the marker does not lie along the y=0 or x=0 axes?
  double const xyOffnessFactor =
      cos(0.08 * std::abs(dirToOrigin.x * dirToOrigin.y) *
          cos(cv::norm(fromCenter) / cv::norm(imageCenter)));
  // double const xyOffnessFactor =
  //    cos(0.06825 * abs(dirToOrigin.x * dirToOrigin.y));
  // double const xyOffnessFactor =
  //    cos(0.025 * cv::norm(fromCenter) / cv::norm(imageCenter));
  // double const xyOffnessFactor = 1.0;
  //
  // detectorEllipsenessInclusion:
  // If a circle has been stretched out by a scaling factor x along one axis,
  // and turned into an ellipse, how large part of that elongation does
  // the SimpleBlobDetector include/enclose in its marker size?
  double constexpr detectorEllipsenessInclusion{0.5};

  double const semiMajorAxis = (blob.size / 2.0) * xyOffnessFactor;
  double const majorAxis = 2 * semiMajorAxis;

  // Step 2: We know that this ellipsis is a projection cast by a circular
  // disc
  //         which is the part of the spherical marker that is facing towards
  //         the pinhole.
  //         The facing disc's normal points directly towards the pinhole.
  //
  //         The ray from it that is closest to the image center,
  //         and the ray that is farthest from the image center,
  //         have entered the pinhole at two deducable angles
  //         (angles measured from z-axis to ray)
  PixelPosition const closestPoint = fromCenter + semiMajorAxis * dirToOrigin;
  PixelPosition const farthestPoint = fromCenter - semiMajorAxis * dirToOrigin;
  double smallestAng = atan(cv::norm(closestPoint) / focalLength);
  double const largestAng = atan(cv::norm(farthestPoint) / focalLength);
  if (cv::norm(fromCenter) < semiMajorAxis) {
    smallestAng = -smallestAng;
  }
  // facing disc's midpoint ang
  double const alpha = std::midpoint(smallestAng, largestAng);
  // facing disc's angular radius seen from the pinhole
  double const gamma = std::midpoint(largestAng, -smallestAng);

  // The center point of the facing disc is not projected onto
  // the center of the ellipsis we see on the sensor
  // Rather, we need to go via alpha to find that
  auto projectionOfFacingDiscCenterPoint{-dirToOrigin * focalLength *
                                         tan(alpha)};

  // The facing disc is not parallell to the image plane,
  // therefore, its projection gets dragged out into an ellipse
  double const ellipsisFactor =
      cos(gamma) * (0.5 / cos(alpha + gamma) + 0.5 / cos(alpha - gamma));

  // This alpha and gamma are correct for the ellipsis that the detector
  // gave us. But did it account for the ellipsis being non-spherical?
  // If detectorEllipsenessInclusion == 1 then yes, otherwise, we need
  // to correct alpha and gamma before continuing

  // Unless detectorEllipsenessInclusion was exactly 1, we've
  // worked with the wrong majorAxis, which have led us to the wrong
  // gamma. Let's create a corrected gamma
  double const correctedEllipsisFactor =
      (1 - detectorEllipsenessInclusion) +
      detectorEllipsenessInclusion * ellipsisFactor;

  // This corrected gamma is still not perfect, but it's about as good
  // as we'll be able to make it with incomplete data
  double const correctedGamma = gamma * correctedEllipsisFactor;

  // Step 3: How would the facing disc project onto the sensor?
  // The facing disc's diameter will be slightly smaller than
  // the full sphere's diameter
  double const facingDiscDiameter = markerDiameter * cos(correctedGamma);

  // It's center will also be slightly in front of the sphere's center.

  // If it was an ellipse, with the same angular center as the facing disc,
  // but standing parallell with the image plane, that cast the projection
  // onto the sensor, then it would need to have had the following width
  // along its semi-major axis
  double const imaginaryEllipsisWidth =
      facingDiscDiameter * correctedEllipsisFactor;

  // Distance to facing disc
  double const z0 = focalLength * (imaginaryEllipsisWidth / (majorAxis));
  double const x0 = projectionOfFacingDiscCenterPoint.x *
                    (imaginaryEllipsisWidth / majorAxis);
  double const y0 = projectionOfFacingDiscCenterPoint.y *
                    (imaginaryEllipsisWidth / majorAxis);

  CameraFramedPosition const P{x0, y0, z0};

  // Distance from pinhole to the facingDisc is:
  double const b0 = cv::norm(P);

  // Distance from facing discmarker's physical center is
  double const b1 = (markerDiameter / 2) * sin(correctedGamma);
  // double const totalDist = b0 + b1;

  // Correcting for the offset between sphere center and facing center
  // finally gives us the real center point
  return P * (1 + b1 / b0);
}
