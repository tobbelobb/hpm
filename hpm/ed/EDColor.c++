#include <hpm/ed/ED.h++>
#include <hpm/ed/EDColor.h++>
#include <hpm/ed/EDCommon.h++>

#include <numeric>

using namespace cv;
using namespace std;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
EDColor::EDColor(const Mat &srcImage, EDColorConfig const &config) {
  int gradThresh = std::max(config.gradThresh, 1);
  int anchorThresh = std::max(config.anchorThresh, 0);
  double blurSize = std::max(config.blurSize, 1.0);

  if (config.filterSegments) {
    anchorThresh = 0;
  }

  height = srcImage.rows;
  width = srcImage.cols;

  auto lab = MyRGB2LabFast(srcImage);
  Mat labClone;
  if (config.filterSegments) {
    labClone = lab.clone();
  }
  blur(lab, blurSize);
  auto const [gradImage, dirData] = ComputeGradientMapByDiZenzo(lab);

  if (config.filterSegments) {
    ED edgeObj = ED(gradImage, dirData, gradThresh, anchorThresh, 1, false);
    segments = edgeObj.getSegments();

    blur(labClone, blurSize / 2.5);
    edgeImage = makeEdgeImage(labClone);
    segments = validSegments(edgeImage, segments);
  } else {
    ED edgeObj = ED(gradImage, dirData, gradThresh, anchorThresh);
    segments = edgeObj.getSegments();
    edgeImage = edgeObj.getEdgeImage();
  }
}

template <std::size_t N> auto static getBgrLut() -> std::array<double, N + 1> {
  std::array<double, N + 1> LUT{};

  for (size_t i = 0; i < N + 1; ++i) {
    double const d = static_cast<double>(i) / static_cast<double>(N);
    if (d >= 0.04045) {
      LUT[i] = pow(((d + 0.055) / 1.055), 2.4);
    } else {
      LUT[i] = d / 12.92;
    }
  }

  return LUT;
}

template <std::size_t N> auto static getXyzLut() -> std::array<double, N + 1> {
  std::array<double, N + 1> LUT{};

  for (size_t i = 0; i < N + 1; ++i) {
    double const d = static_cast<double>(i) / static_cast<double>(N);
    if (d > 0.008856) {
      LUT[i] = pow(d, 1.0 / 3.0);
    } else {
      LUT[i] = (7.787 * d) + (16.0 / 116.0);
    }
  }

  return LUT;
}

auto EDColor::MyRGB2LabFast(cv::Mat srcImage) -> cv::Mat {
  static size_t constexpr LUT_SIZE{1024 * 4096};

  auto const size{static_cast<size_t>(width * height)};
  std::vector<double> L(size, 0.0);
  std::vector<double> a(size, 0.0);
  std::vector<double> b(size, 0.0);

  std::array<cv::Mat, 3> bgr;
  split(srcImage, bgr);

  // Get rgb->xyz->lab values
  // xyz observer = 2deg, illuminant = D65
  srcImage.forEach<Point3_<uint8_t>>([&](auto &point, int const positions[]) {
    auto const pos = static_cast<size_t>(positions[0] * width + positions[1]);
    Point3d bgr{point.x / 255.0, point.y / 255.0, point.z / 255.0};
    static auto const BGR_LUT = getBgrLut<LUT_SIZE>();
    bgr = {100 * BGR_LUT[static_cast<size_t>(bgr.x * LUT_SIZE + 0.5)],
           100 * BGR_LUT[static_cast<size_t>(bgr.y * LUT_SIZE + 0.5)],
           100 * BGR_LUT[static_cast<size_t>(bgr.z * LUT_SIZE + 0.5)]};
    double x =
        (bgr.z * 0.4124564 + bgr.y * 0.3575761 + bgr.x * 0.1804375) / 95.047;
    double y =
        (bgr.z * 0.2126729 + bgr.y * 0.7151522 + bgr.x * 0.0721750) / 100.000;
    double z =
        (bgr.z * 0.0193339 + bgr.y * 0.1191920 + bgr.x * 0.9503041) / 108.883;
    static auto const XYZ_LUT = getXyzLut<LUT_SIZE>();
    x = XYZ_LUT[static_cast<size_t>(x * LUT_SIZE + 0.5)];
    y = XYZ_LUT[static_cast<size_t>(y * LUT_SIZE + 0.5)];
    z = XYZ_LUT[static_cast<size_t>(z * LUT_SIZE + 0.5)];
    L[pos] = ((116.0 * y) - 16);
    a[pos] = (500 * (x / y));
    b[pos] = (200 * (y - z));
  });

  cv::Mat Lab_Img(height, width, LAB_PIX_CV_TYPE);

  auto const [minLiter, maxLiter] = std::minmax_element(L.begin(), L.end());
  auto const [minAiter, maxAiter] = std::minmax_element(a.begin(), a.end());
  auto const [minBiter, maxBiter] = std::minmax_element(b.begin(), b.end());
  auto const minL{*minLiter};
  auto const minA{*minAiter};
  auto const minB{*minBiter};
  double const scaleL = 255.0 / (*maxLiter - minL);
  double const scaleA = 255.0 / (*maxAiter - minA);
  double const scaleB = 255.0 / (*maxBiter - minB);

  Lab_Img.forEach<LabPix>([&](auto &point, int const positions[]) {
    auto const pos = static_cast<size_t>(positions[0] * width + positions[1]);
    point = {static_cast<LabPixSingle>((L[pos] - minL) * scaleL),
             static_cast<LabPixSingle>((a[pos] - minA) * scaleA),
             static_cast<LabPixSingle>((b[pos] - minB) * scaleB)};
  });

  return Lab_Img;
}

auto EDColor::ComputeGradientMapByDiZenzo(cv::Mat lab) -> GradientMapResult {

  std::vector<EdgeDir> dirData(static_cast<size_t>(width * height),
                               EdgeDir::NONE);
  cv::Mat gradImage(height, width, GRAD_PIX_CV_TYPE, cv::Scalar(0));

  auto *labPtr = lab.ptr<LabPix>(0);
  auto *gradImg = gradImage.ptr<GradPix>(0);
  cv::Mat frameless(lab, cv::Rect(1, 1, width - 2, height - 2));

  int max = 0;
  frameless.forEach<LabPix>([&](auto &unused, const int position[]) {
    (void)unused;
    int const i{position[0] + 1};
    int const j{position[1] + 1};
    Point3i const downRight{labPtr[(i + 1) * width + j + 1]};
    Point3i const upLeft{labPtr[(i - 1) * width + j - 1]};
    Point3i const upRight{labPtr[(i - 1) * width + j + 1]};
    Point3i const downLeft{labPtr[(i + 1) * width + j - 1]};
    Point3i const com1 = downRight - upLeft;
    Point3i const com2 = upRight - downLeft;

    LabPix const &right{labPtr[i * width + j + 1]};
    LabPix const &left{labPtr[i * width + j - 1]};
    LabPix const &down{labPtr[(i + 1) * width + j]};
    LabPix const &up{labPtr[(i - 1) * width + j]};

    int const gx0 = com1.x + com2.x + right.x - left.x;
    int const gx1 = com1.y + com2.y + right.y - left.y;
    int const gx2 = com1.z + com2.z + right.z - left.z;

    int const gy0 = com1.x + down.x - com2.x - up.x;
    int const gy1 = com1.y + down.y - com2.y - up.y;
    int const gy2 = com1.z + down.z - com2.z - up.z;

    int const gxx = gx0 * gx0 + gx1 * gx1 + gx2 * gx2;
    int const gyy = gy0 * gy0 + gy1 * gy1 + gy2 * gy2;
    int const gxy = gx0 * gy0 + gx1 * gy1 + gx2 * gy2;

    // Di Zenzo's formulas from Gonzales & Woods - Page 337
    double twoTheta =
        atan2(2.0 * gxy, static_cast<double>(gxx - gyy)); // Gradient Direction
    auto grad =
        static_cast<GradPix>(sqrt((gxx + gyy + (gxx - gyy) * cos(twoTheta) +
                                   2 * gxy * sin(twoTheta)) /
                                  2.0) +
                             0.5); // Gradient Magnitude

    // Gradient is perpendicular to the edge passing through the pixel
    if (twoTheta >= -CV_PI / 2.0 and twoTheta <= CV_PI / 2.0) {
      dirData[i * width + j] = EdgeDir::VERTICAL;
    } else {
      dirData[i * width + j] = EdgeDir::HORIZONTAL;
    }

    gradImg[i * width + j] = grad;
    if (grad > max) {
      max = grad;
    }
  });

  // Scale the gradient values to 0-255
  double const scale = 255.0 / max;
  gradImage.forEach<GradPix>([&](GradPix &pixel, int const unused[]) {
    (void)unused;
    pixel = static_cast<GradPix>(static_cast<double>(pixel) * scale);
  });

  return {gradImage, dirData};
}

void EDColor::blur(cv::Mat src, double const blurSize) {
  auto getBlurConfig = [](double const blurSize) -> std::pair<Size, double> {
    if (blurSize >= 1.0 and blurSize < 1.5) {
      return {Size(5, 5), blurSize};
    }
    if (blurSize >= 1.5) {
      return {Size(7, 7), blurSize};
    }
    return {Size(), blurSize};
  };

  auto const [kernelSize, sigma] = getBlurConfig(blurSize);

  GaussianBlur(src, src, kernelSize, sigma);
}

//--------------------------------------------------------------------------------------------------------------------
// Filter edges using the Helmholtz principle
// Create a gradient image based on Lab encoded input image
// Then redraw the edgeImage based on the new gradient image
auto EDColor::makeEdgeImage(cv::Mat lab) -> cv::Mat {

  cv::Mat gradImage(height, width, GRAD_PIX_CV_TYPE, Scalar(0));
  cv::Mat edgeImageOut(height, width, CV_8UC1, Scalar(0));

  auto const *labPtr = lab.ptr<LabPix>(0);
  auto *gradImg = gradImage.ptr<GradPix>(0);
  cv::Mat frameless(lab, cv::Rect(1, 1, width - 2, height - 2));
  frameless.forEach<LabPix>([&](auto &unused, const int position[]) {
    (void)unused;
    int const i{position[0] + 1};
    int const j{position[1] + 1};
    Point3i const downRight{labPtr[(i + 1) * width + j + 1]};
    Point3i const upLeft{labPtr[(i - 1) * width + j - 1]};
    Point3i const upRight{labPtr[(i - 1) * width + j + 1]};
    Point3i const downLeft{labPtr[(i + 1) * width + j - 1]};
    Point3i const com1 = downRight - upLeft;
    Point3i const com2 = upRight - downLeft;

    LabPix const &currRight{labPtr[i * width + j + 1]};
    LabPix const &currLeft{labPtr[i * width + j - 1]};
    LabPix const &downCurr{labPtr[(i + 1) * width + j]};
    LabPix const &upCurr{labPtr[(i - 1) * width + j]};

    int const gx0 = abs(com1.x + com2.x + currRight.x - currLeft.x);
    int const gx1 = abs(com1.y + com2.y + currRight.y - currLeft.y);
    int const gx2 = abs(com1.z + com2.z + currRight.z - currLeft.z);

    int const gy0 = abs(com1.x + downCurr.x - com2.x - upCurr.x);
    int const gy1 = abs(com1.y + downCurr.y - com2.y - upCurr.y);
    int const gy2 = abs(com1.z + downCurr.z - com2.z - upCurr.z);

    auto const grad =
        static_cast<GradPix>(((gx0 + gx1 + gx2 + gy0 + gy1 + gy2) + 2) / 3);
    gradImg[i * width + j] = grad;
  });

  std::vector<int> grads(MAX_GRAD_VALUE, 0);
  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      grads[static_cast<size_t>(gradImg[i * width + j])]++;
    }
  }
  for (size_t i = MAX_GRAD_VALUE - 1; i > 0; i--) {
    grads[i - 1] += grads[i];
  }

  std::vector<double> probabilityFunctionH(MAX_GRAD_VALUE, 0.0);
  for (size_t i = 0; i < MAX_GRAD_VALUE; i++) {
    probabilityFunctionH[i] =
        grads[i] / static_cast<double>(frameless.rows * frameless.cols);
  }

  int const numberOfSegmentPieces = std::transform_reduce(
      segments.begin(), segments.end(), 0, std::plus<>(),
      [](auto const &segment) {
        return (segment.size() * (segment.size() - 1)) / 2;
      });

  for (auto const &segment : segments) {
    drawFilteredSegment(segment.begin(), segment.end(), edgeImageOut, gradImage,
                        probabilityFunctionH, numberOfSegmentPieces);
  }
  return edgeImageOut;
}
#pragma GCC diagnostic pop
