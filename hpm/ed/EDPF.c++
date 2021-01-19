#include <numeric>

#include <hpm/ed/EDCommon.h++>
#include <hpm/ed/EDPF.h++>
#include <hpm/ed/EDTypes.h++>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
using namespace cv;
using namespace std;

EDPF::EDPF(const Mat &srcImage)
    : ED(srcImage, {.op = GradientOperator::PREWITT,
                    .gradThresh = 11,
                    .anchorThresh = 3}) {
  blurSize /= 2.5;
  GaussianBlur(srcImage, smoothImage, Size(), blurSize);
  edgeImage = makeEdgeImage();
  segments = validSegments(edgeImage, segments);
}

EDPF::EDPF(const ED &obj) : ED(obj) {
  blurSize /= 2.5;
  GaussianBlur(srcImage, smoothImage, Size(), blurSize);
  edgeImage = makeEdgeImage();
  segments = validSegments(edgeImage, segments);
}

EDPF::EDPF(const EDColor &obj) : ED(obj) {}

auto EDPF::makeEdgeImage() -> cv::Mat {
  auto [gradImage, probabilityFunctionH] = ComputePrewitt3x3();

  int const numberOfSegmentPieces = std::transform_reduce(
      segments.begin(), segments.end(), 0, std::plus<>(),
      [](auto const &segment) {
        return (segment.size() * (segment.size() - 1)) / 2;
      });

  cv::Mat edgeImageOut = Mat::zeros(height, width, CV_8UC1);
  for (auto const &segment : segments) {
    drawFilteredSegment(segment.begin(), segment.end(), edgeImageOut, gradImage,
                        probabilityFunctionH, numberOfSegmentPieces);
  }
  return edgeImageOut;
}

auto EDPF::ComputePrewitt3x3()
    -> std::pair<cv::Mat_<GradPix>, std::vector<double>> {
  cv::Mat gradImage(height, width, GRAD_PIX_CV_TYPE, cv::Scalar(0));

  auto const *srcImg = srcImage.ptr<uint8_t>(0);
  auto *gradImg = gradImage.ptr<GradPix>(0);
  cv::Mat frameless(srcImage, cv::Rect(1, 1, width - 2, height - 2));
  frameless.forEach<uint8_t>([&](auto &unused, const int position[]) {
    (void)unused;
    int const i{position[0] + 1};
    int const j{position[1] + 1};
    GradPix const downRight{srcImg[(i + 1) * width + j + 1]};
    GradPix const upLeft{srcImg[(i - 1) * width + j - 1]};
    GradPix const upRight{srcImg[(i - 1) * width + j + 1]};
    GradPix const downLeft{srcImg[(i + 1) * width + j - 1]};
    GradPix const currRight{srcImg[i * width + j + 1]};
    GradPix const currLeft{srcImg[i * width + j - 1]};
    GradPix const downCurr{srcImg[(i + 1) * width + j]};
    GradPix const upCurr{srcImg[(i - 1) * width + j]};
    // Prewitt Operator in horizontal and vertical direction
    GradPix const com1 = downRight - upLeft;
    GradPix const com2 = upRight - downLeft;
    auto const gx =
        static_cast<GradPix>(abs(com1 + com2 + currRight - currLeft));
    auto const gy = static_cast<GradPix>(abs(com1 - com2 + downCurr - upCurr));

    gradImg[i * width + j] = gx + gy;
  });

  std::vector<int> grads(MAX_GRAD_VALUE, 0);
  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      grads[static_cast<size_t>(gradImg[i * width + j])]++;
    }
  }
  for (int i = MAX_GRAD_VALUE - 1; i > 0; i--) {
    grads[i - 1] += grads[i];
  }

  std::vector<double> probabilityFunctionH(MAX_GRAD_VALUE, 0.0);
  for (int i = 0; i < probabilityFunctionH.size(); i++) {
    probabilityFunctionH[i] =
        static_cast<double>(grads[i]) /
        static_cast<double>(frameless.rows * frameless.cols);
  }

  return {gradImage, probabilityFunctionH};
}
#pragma GCC diagnostic pop
