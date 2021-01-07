#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <hpm/ed/EDLib.h++>
#pragma GCC diagnostic pop

#include <hpm/ellipse-detector.h++>
#include <hpm/detection-result.h++>
#include <hpm/util.h++>

using namespace hpm;

auto ellipseDetect(cv::InputArray image, bool showIntermediateImages)
    -> hpm::DetectionResult {
  cv::Mat imageMat{image.getMat()};
  ED testED = ED(imageMat, SOBEL_OPERATOR, 36, 8, 1, 10, 1.0, true);
  cv::Mat edgeImage{testED.getEdgeImage()};

  if (showIntermediateImages) {
    showImage(edgeImage, "edgeImage.png");
  }

  return {};
}
