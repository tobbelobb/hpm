#include <numeric>

#include <hpm/ed/EDCommon.h++>
#include <hpm/ed/EDPF.h++>
#include <hpm/ed/EDTypes.h++>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
using namespace cv;
using namespace std;

EDPF::EDPF(Mat srcImage)
    : ED(srcImage, {.op = GradientOperator::PREWITT,
                    .gradThresh = 11,
                    .anchorThresh = 3}) {
  blurSize /= 2.5;
  GaussianBlur(srcImage, smoothImage, Size(), blurSize);
  edgeImage = makeEdgeImage();
  segments = validSegments(edgeImage, segments);
}

EDPF::EDPF(ED obj) : ED(obj) {
  blurSize /= 2.5;
  GaussianBlur(srcImage, smoothImage, Size(), blurSize);
  edgeImage = makeEdgeImage();
  segments = validSegments(edgeImage, segments);
}

EDPF::EDPF(EDColor obj) : ED(obj) {}

cv::Mat EDPF::makeEdgeImage() {
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

std::pair<cv::Mat_<GradPix>, std::vector<double>> EDPF::ComputePrewitt3x3() {
  cv::Mat gradImage(height, width, GRAD_PIX_CV_TYPE, cv::Scalar(0));

  auto const *srcImg = srcImage.ptr<uint8_t>(0);
  auto *gradImg = gradImage.ptr<GradPix>(0);
  cv::Mat frameless(srcImage, cv::Rect(1, 1, width - 2, height - 2));
  frameless.forEach<uint8_t>([&](auto &pix, const int position[]) {
    int const i{position[0] + 1};
    int const j{position[1] + 1};
    int const downRight{srcImg[(i + 1) * width + j + 1]};
    int const upLeft{srcImg[(i - 1) * width + j - 1]};
    int const upRight{srcImg[(i - 1) * width + j + 1]};
    int const downLeft{srcImg[(i + 1) * width + j - 1]};
    int const currRight{srcImg[i * width + j + 1]};
    int const currLeft{srcImg[i * width + j - 1]};
    int const downCurr{srcImg[(i + 1) * width + j]};
    int const upCurr{srcImg[(i - 1) * width + j]};
    // Prewitt Operator in horizontal and vertical direction
    int const com1 = downRight - upLeft;
    int const com2 = upRight - downLeft;
    int const gx = abs(com1 + com2 + currRight - currLeft);
    int const gy = abs(com1 - com2 + downCurr - upCurr);

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
