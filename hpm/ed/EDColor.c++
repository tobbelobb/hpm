#include <hpm/ed/ED.h++>
#include <hpm/ed/EDColor.h++>

using namespace cv;
using namespace std;

EDColor::EDColor(Mat srcImage, EDColorConfig const &config) {
  double blurSize = std::max(config.blurSize, 1.0);
  double gradThresh = std::max(config.gradThresh, 1);
  double anchorThresh = std::max(config.anchorThresh, 0);

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
    ED edgeObj = ED(gradImage, dirData, gradThresh, anchorThresh, 1, 10, false);
    segments = edgeObj.getSegments();

    blur(labClone, blurSize / 2.5);
    edgeImage = makeEdgeImage(labClone);
    segments = validSegments(edgeImage, segments);
  } else {
    ED edgeObj = ED(gradImage, dirData, gradThresh, anchorThresh);
    segments = edgeObj.getSegments();
    edgeImage = edgeObj.getEdgeImage();
  }

  // Fix 1 pixel errors in the edge map
  fixEdgeSegments(segments, 1);
}

cv::Mat EDColor::getEdgeImage() { return edgeImage; }

std::vector<std::vector<cv::Point>> EDColor::getSegments() const {
  return segments;
}

int EDColor::getNumberOfSegments() const { return segments.size(); }

int EDColor::getWidth() const { return width; }

int EDColor::getHeight() const { return height; }

template <std::size_t N> std::array<double, N + 1> static getLut(int which) {
  std::array<double, N + 1> LUT;

  for (size_t i = 0; i < N + 1; ++i) {
    double const d = static_cast<double>(i) / static_cast<double>(N);
    if (which == 1) {
      if (d >= 0.04045) {
        LUT[i] = pow(((d + 0.055) / 1.055), 2.4);
      } else {
        LUT[i] = d / 12.92;
      }
    } else {
      if (d > 0.008856) {
        LUT[i] = pow(d, 1.0 / 3.0);
      } else {
        LUT[i] = (7.787 * d) + (16.0 / 116.0);
      }
    }
  }

  return LUT;
}

cv::Mat EDColor::MyRGB2LabFast(cv::Mat srcImage) {
  static size_t constexpr LUT_SIZE{1024 * 4096};
  static auto const LUT1 = getLut<LUT_SIZE>(1);
  static auto const LUT2 = getLut<LUT_SIZE>(2);

  auto const size{static_cast<size_t>(width * height)};
  std::vector<double> L(size, 0.0);
  std::vector<double> a(size, 0.0);
  std::vector<double> b(size, 0.0);

  std::array<cv::Mat, 3> bgr;
  split(srcImage, bgr);

  // Get rgb->xyz->lab values
  // xyz observer = 2deg, illuminant = D65
  srcImage.forEach<Point3_<uchar>>([&](auto &point, int const positions[]) {
    int const pos = positions[0] * width + positions[1];
    Point3d bgr{point.x / 255.0, point.y / 255.0, point.z / 255.0};
    bgr = {100 * LUT1[(int)(bgr.x * LUT_SIZE + 0.5)],
           100 * LUT1[(int)(bgr.y * LUT_SIZE + 0.5)],
           100 * LUT1[(int)(bgr.z * LUT_SIZE + 0.5)]};
    double x =
        (bgr.z * 0.4124564 + bgr.y * 0.3575761 + bgr.x * 0.1804375) / 95.047;
    double y =
        (bgr.z * 0.2126729 + bgr.y * 0.7151522 + bgr.x * 0.0721750) / 100.000;
    double z =
        (bgr.z * 0.0193339 + bgr.y * 0.1191920 + bgr.x * 0.9503041) / 108.883;
    x = LUT2[(int)(x * LUT_SIZE + 0.5)];
    y = LUT2[(int)(y * LUT_SIZE + 0.5)];
    z = LUT2[(int)(z * LUT_SIZE + 0.5)];
    L[pos] = ((116.0 * y) - 16);
    a[pos] = (500 * (x / y));
    b[pos] = (200 * (y - z));
  });

  // for (int i = 0; i < size; i++) {
  //  // RGB to XYZ
  //  double red = bgr[2].data[i] / 255.0;
  //  double green = bgr[1].data[i] / 255.0;
  //  double blue = bgr[0].data[i] / 255.0;

  //  red = LUT1[(int)(red * LUT_SIZE + 0.5)];
  //  green = LUT1[(int)(green * LUT_SIZE + 0.5)];
  //  blue = LUT1[(int)(blue * LUT_SIZE + 0.5)];

  //  red = red * 100;
  //  green = green * 100;
  //  blue = blue * 100;

  //  // Observer. = 2deg, Illuminant = D65
  //  double x = red * 0.4124564 + green * 0.3575761 + blue * 0.1804375;
  //  double y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750;
  //  double z = red * 0.0193339 + green * 0.1191920 + blue * 0.9503041;

  //  // Now xyz 2 Lab
  //  double refX = 95.047;
  //  double refY = 100.000;
  //  double refZ = 108.883;

  //  x = x / refX; // ref_X =  95.047   Observer= 2deg, Illuminant= D65
  //  y = y / refY; // ref_Y = 100.000
  //  z = z / refZ; // ref_Z = 108.883

  //  x = LUT2[(int)(x * LUT_SIZE + 0.5)];
  //  y = LUT2[(int)(y * LUT_SIZE + 0.5)];
  //  z = LUT2[(int)(z * LUT_SIZE + 0.5)];

  //  L[i] = ((116.0 * y) - 16);
  //  a[i] = (500 * (x / y));
  //  b[i] = (200 * (y - z));
  //}

  cv::Mat Lab_Img(height, width, LAB_PIX_CV_TYPE);

  auto const [minL, maxL] = std::minmax_element(L.begin(), L.end());
  auto const [minA, maxA] = std::minmax_element(a.begin(), a.end());
  auto const [minB, maxB] = std::minmax_element(b.begin(), b.end());
  double const scaleL = 255.0 / (*maxL - *minL);
  double const scaleA = 255.0 / (*maxA - *minA);
  double const scaleB = 255.0 / (*maxB - *minB);

  Lab_Img.forEach<LabPix>([&](auto &point, int const positions[]) {
    int const pos = positions[0] * width + positions[1];
    point = {static_cast<LabPixSingle>((L[pos] - *minL) * scaleL),
             static_cast<LabPixSingle>((a[pos] - *minA) * scaleA),
             static_cast<LabPixSingle>((b[pos] - *minB) * scaleB)};
  });

  return Lab_Img;
}

GradientMapResult EDColor::ComputeGradientMapByDiZenzo(cv::Mat lab) {

  std::array<cv::Mat, 3> labSplit;
  split(lab, labSplit);

  cv::Mat l = labSplit[0];
  cv::Mat a = labSplit[1];
  cv::Mat b = labSplit[2];
  std::vector<EdgeDir> dirData{};
  dirData.resize(height * width);
  std::fill(dirData.begin(), dirData.end(), EdgeDir::NONE);
  cv::Mat gradImage(height, width, GRAD_PIX_CV_TYPE, cv::Scalar(0));

  int max = 0;

  GradPix *gradImg = gradImage.ptr<GradPix>(0);
  auto *lPtr = l.ptr<LabPixSingle>(0);
  auto *aPtr = a.ptr<LabPixSingle>(0);
  auto *bPtr = b.ptr<LabPixSingle>(0);
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      // Prewitt for channel1
      int com1 = lPtr[(i + 1) * width + j + 1] - lPtr[(i - 1) * width + j - 1];
      int com2 = lPtr[(i - 1) * width + j + 1] - lPtr[(i + 1) * width + j - 1];

      int gxCh1 =
          com1 + com2 + (lPtr[i * width + j + 1] - lPtr[i * width + j - 1]);
      int gyCh1 =
          com1 - com2 + (lPtr[(i + 1) * width + j] - lPtr[(i - 1) * width + j]);

      // Prewitt for channel2
      com1 = aPtr[(i + 1) * width + j + 1] - aPtr[(i - 1) * width + j - 1];
      com2 = aPtr[(i - 1) * width + j + 1] - aPtr[(i + 1) * width + j - 1];

      int gxCh2 =
          com1 + com2 + (aPtr[i * width + j + 1] - aPtr[i * width + j - 1]);
      int gyCh2 =
          com1 - com2 + (aPtr[(i + 1) * width + j] - aPtr[(i - 1) * width + j]);

      // Prewitt for channel3
      com1 = bPtr[(i + 1) * width + j + 1] - bPtr[(i - 1) * width + j - 1];
      com2 = bPtr[(i - 1) * width + j + 1] - bPtr[(i + 1) * width + j - 1];

      int gxCh3 =
          com1 + com2 + (bPtr[i * width + j + 1] - bPtr[i * width + j - 1]);
      int gyCh3 =
          com1 - com2 + (bPtr[(i + 1) * width + j] - bPtr[(i - 1) * width + j]);
      int gxx = gxCh1 * gxCh1 + gxCh2 * gxCh2 + gxCh3 * gxCh3;
      int gyy = gyCh1 * gyCh1 + gyCh2 * gyCh2 + gyCh3 * gyCh3;
      int gxy = gxCh1 * gyCh1 + gxCh2 * gyCh2 + gxCh3 * gyCh3;

      // Di Zenzo's formulas from Gonzales & Woods - Page 337
      double theta =
          atan2(2.0 * gxy, (double)(gxx - gyy)) / 2; // Gradient Direction
      int grad = (int)(sqrt(((gxx + gyy) + (gxx - gyy) * cos(2 * theta) +
                             2 * gxy * sin(2 * theta)) /
                            2.0) +
                       0.5); // Gradient Magnitude

      // Gradient is perpendicular to the edge passing through the pixel
      if (theta >= -3.14159 / 4 && theta <= 3.14159 / 4)
        dirData[i * width + j] = EdgeDir::VERTICAL;
      else
        dirData[i * width + j] = EdgeDir::HORIZONTAL;

      gradImg[i * width + j] = grad;
      if (grad > max)
        max = grad;
    }
  } // end outer for

  // Scale the gradient values to 0-255
  double const scale = 255.0 / max;

  gradImage.forEach<GradPix>([scale](GradPix &pixel, int const positions[]) {
    pixel = (GradPix)(pixel * scale);
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
cv::Mat EDColor::makeEdgeImage(cv::Mat lab) {
  int maxGradValue = MAX_GRAD_VALUE;
  std::vector<double> probabilityFunctionH(maxGradValue, 0.0);

  cv::Mat_<GradPix> gradImage = Mat::zeros(height, width, GRAD_PIX_CV_TYPE);
  cv::Mat edgeImageOut = Mat::zeros(height, width, CV_8UC1);

  std::vector<int> grads(maxGradValue, 0);

  auto *labPtr = lab.ptr<LabPix>(0);
  auto *gradImg = gradImage.ptr<GradPix>(0);

  cv::Mat frameless(lab, cv::Rect(1, 1, width - 2, height - 2));
  frameless.forEach<LabPix>([&](auto &point, const int position[]) {
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

    GradPix grad =
        (static_cast<GradPix>(gx0 + gx1 + gx2 + gy0 + gy1 + gy2) + 2) / 3;
    gradImg[i * width + j] = grad;
    grads[grad]++;
  });

  // Compute probability function H
  int size = (width - 2) * (height - 2);

  for (int i = maxGradValue - 1; i > 0; i--)
    grads[i - 1] += grads[i];

  for (int i = 0; i < maxGradValue; i++)
    probabilityFunctionH[i] = (double)grads[i] / ((double)size);

  int numberOfSegmentPieces = 0;
  for (int i = 0; i < segments.size(); i++) {
    int len = segments[i].size();
    numberOfSegmentPieces += (len * (len - 1)) / 2;
  }

  // Validate segments
  for (auto const &segment : segments) {
    drawFilteredSegment(segment.begin(), segment.end(), edgeImageOut, gradImage,
                        probabilityFunctionH, numberOfSegmentPieces);
  }
  return edgeImageOut;
}

static double NFA(double prob, int len, int const numberOfSegmentPieces) {
  double nfa = static_cast<double>(numberOfSegmentPieces);
  for (int i = 0; i < len && nfa > EPSILON; i++) {
    nfa *= prob;
  }
  return nfa;
}

//----------------------------------------------------------------------------------
// Resursive validation using half of the pixels as suggested by DMM algorithm
// We take pixels at Nyquist distance, i.e., 2 (as suggested by DMM)
//
template <typename Iterator>
void EDColor::drawFilteredSegment(
    Iterator firstPoint, Iterator lastPoint, cv::Mat edgeImageIn,
    cv::Mat_<GradPix> gradImage,
    std::vector<double> const &probabilityFunctionH,
    int const numberOfSegmentPieces) {

  int const chainLen = std::distance(firstPoint, lastPoint);
  if (chainLen < MIN_SEGMENT_LEN) {
    return;
  }

  // First find the min. gradient along the segment
  GradPix const *gradImg = gradImage.ptr<GradPix>(0);
  auto minGradPoint = std::min_element(
      firstPoint, lastPoint, [&](Point const &p0, Point const &p1) {
        return gradImg[p0.y * width + p0.x] < gradImg[p1.y * width + p1.x];
      });
  int minGrad = gradImg[(*minGradPoint).y * width + (*minGradPoint).x];

  double nfa = NFA(probabilityFunctionH[minGrad], (int)(chainLen / 2.25),
                   numberOfSegmentPieces);

  // Draw subsegment on edgeImage
  uchar *edgeImg = edgeImageIn.ptr<uchar>(0);
  if (nfa <= EPSILON) {
    std::for_each(firstPoint, lastPoint, [&](auto const &point) {
      edgeImg[point.y * width + point.x] = 255;
    });

    return;
  }

  drawFilteredSegment(firstPoint, minGradPoint, edgeImageIn, gradImage,
                      probabilityFunctionH, numberOfSegmentPieces);
  drawFilteredSegment(std::next(minGradPoint), lastPoint, edgeImageIn,
                      gradImage, probabilityFunctionH, numberOfSegmentPieces);
}

vector<Segment> EDColor::validSegments(cv::Mat_<uchar> edgeImageIn,
                                       vector<Segment> segmentsIn) const {
  vector<Segment> valids;

  uchar *edgeImg = edgeImageIn.ptr<uchar>(0);
  for (auto const &segment : segmentsIn) {
    auto const end = segment.end();
    auto front = segment.begin();
    auto back = segment.begin();
    while (back != end) {
      front = std::find_if(front, end, [&](auto const &point) {
        return edgeImg[point.y * width + point.x];
      });
      back = std::find_if_not(front, end, [&](auto const &point) {
        return edgeImg[point.y * width + point.x];
      });

      if (std::distance(front, back) >= MIN_SEGMENT_LEN) {
        valids.emplace_back(front, std::prev(back));
      }
      front = std::next(back);
    }
  }

  return valids;
}

//---------------------------------------------------------
// Fix edge segments having one or two pixel fluctuations
// An example one pixel problem getting fixed:
//  x
// x x --> xxx
//
// An example two pixel problem getting fixed:
//  xx
// x  x --> xxxx
//
void EDColor::fixEdgeSegments(std::vector<std::vector<cv::Point>> map,
                              int noPixels) {
  /// First fix one pixel problems: There are four cases
  for (int i = 0; i < map.size(); i++) {
    int cp = map[i].size() - 2; // Current pixel index
    int n2 = 0;                 // next next pixel index

    while (n2 < map[i].size()) {
      int n1 = cp + 1; // next pixel

      cp = cp % map[i].size(); // Roll back to the beginning
      n1 = n1 % map[i].size(); // Roll back to the beginning

      int r = map[i][cp].y;
      int c = map[i][cp].x;

      int r1 = map[i][n1].y;
      int c1 = map[i][n1].x;

      int r2 = map[i][n2].y;
      int c2 = map[i][n2].x;

      // 4 cases to fix
      if (r2 == r - 2 && c2 == c) {
        if (c1 != c) {
          map[i][n1].x = c;
        }

        cp = n2;
        n2 += 2;

      } else if (r2 == r + 2 && c2 == c) {
        if (c1 != c) {
          map[i][n1].x = c;
        }

        cp = n2;
        n2 += 2;

      } else if (r2 == r && c2 == c - 2) {
        if (r1 != r) {
          map[i][n1].y = r;
        }

        cp = n2;
        n2 += 2;

      } else if (r2 == r && c2 == c + 2) {
        if (r1 != r) {
          map[i][n1].y = r;
        }

        cp = n2;
        n2 += 2;

      } else {
        cp++;
        n2++;
      }
    }
  }
}
