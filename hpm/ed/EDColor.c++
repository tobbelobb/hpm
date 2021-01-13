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

  auto LabImgs = MyRGB2LabFast(srcImage);

  auto const [gradImage, dirData] =
      ComputeGradientMapByDiZenzo(blur(LabImgs, blurSize));

  if (config.filterSegments) {
    // Get Edge Image using ED
    ED edgeObj = ED(gradImage, dirData, gradThresh, anchorThresh, 1, 10, false);
    segments = edgeObj.getSegments();
    edgeImage = edgeObj.getEdgeImage();

    blurSize /= 2.5;

    redrawEdgeImage(blur(LabImgs, blurSize));
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

std::array<cv::Mat, 3> EDColor::MyRGB2LabFast(cv::Mat srcImage) {
  static size_t constexpr LUT_SIZE{1024 * 4096};
  static auto const LUT1 = getLut<LUT_SIZE>(1);
  static auto const LUT2 = getLut<LUT_SIZE>(2);

  std::vector<double> L{};
  std::vector<double> a{};
  std::vector<double> b{};
  auto const size{static_cast<size_t>(width * height)};
  L.reserve(size);
  a.reserve(size);
  b.reserve(size);

  std::array<cv::Mat, 3> bgr;
  split(srcImage, bgr);

  for (int i = 0; i < width * height; i++) {
    // RGB to XYZ
    double red = bgr[2].data[i] / 255.0;
    double green = bgr[1].data[i] / 255.0;
    double blue = bgr[0].data[i] / 255.0;

    red = LUT1[(int)(red * LUT_SIZE + 0.5)];
    green = LUT1[(int)(green * LUT_SIZE + 0.5)];
    blue = LUT1[(int)(blue * LUT_SIZE + 0.5)];

    red = red * 100;
    green = green * 100;
    blue = blue * 100;

    // Observer. = 2deg, Illuminant = D65
    double x = red * 0.4124564 + green * 0.3575761 + blue * 0.1804375;
    double y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750;
    double z = red * 0.0193339 + green * 0.1191920 + blue * 0.9503041;

    // Now xyz 2 Lab
    double refX = 95.047;
    double refY = 100.000;
    double refZ = 108.883;

    x = x / refX; // ref_X =  95.047   Observer= 2deg, Illuminant= D65
    y = y / refY; // ref_Y = 100.000
    z = z / refZ; // ref_Z = 108.883

    x = LUT2[(int)(x * LUT_SIZE + 0.5)];
    y = LUT2[(int)(y * LUT_SIZE + 0.5)];
    z = LUT2[(int)(z * LUT_SIZE + 0.5)];

    L.push_back((116.0 * y) - 16);
    a.push_back(500 * (x / y));
    b.push_back(200 * (y - z));
  }

  cv::Mat L_Img(height, width, CV_8UC1);
  cv::Mat a_Img(height, width, CV_8UC1);
  cv::Mat b_Img(height, width, CV_8UC1);

  auto scale255 = [size](std::vector<double> const &Lab, cv::Mat &Lab_Img) {
    auto const [min, max] = std::minmax_element(Lab.begin(), Lab.end());
    double const scale = 255.0 / (*max - *min);
    for (size_t i = 0; i < size; i++) {
      Lab_Img.data[i] = static_cast<unsigned char>((Lab[i] - *min) * scale);
    }
  };
  scale255(L, L_Img);
  scale255(a, a_Img);
  scale255(b, b_Img);

  return {L_Img, a_Img, b_Img};
}

GradientMapResult
EDColor::ComputeGradientMapByDiZenzo(std::array<cv::Mat, 3> smoothLab) {
  cv::Mat smoothL = smoothLab[0];
  cv::Mat smootha = smoothLab[1];
  cv::Mat smoothb = smoothLab[2];
  std::vector<EdgeDir> dirData{};
  dirData.resize(height * width);
  std::fill(dirData.begin(), dirData.end(), EdgeDir::NONE);
  cv::Mat gradImage(height, width, CV_16SC1, cv::Scalar(0));

  int max = 0;

  GradPix *gradImg = gradImage.ptr<GradPix>(0);
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      // Prewitt for channel1
      int com1 = smoothL.data[(i + 1) * width + j + 1] -
                 smoothL.data[(i - 1) * width + j - 1];
      int com2 = smoothL.data[(i - 1) * width + j + 1] -
                 smoothL.data[(i + 1) * width + j - 1];

      int gxCh1 =
          com1 + com2 +
          (smoothL.data[i * width + j + 1] - smoothL.data[i * width + j - 1]);
      int gyCh1 = com1 - com2 +
                  (smoothL.data[(i + 1) * width + j] -
                   smoothL.data[(i - 1) * width + j]);

      // Prewitt for channel2
      com1 = smootha.data[(i + 1) * width + j + 1] -
             smootha.data[(i - 1) * width + j - 1];
      com2 = smootha.data[(i - 1) * width + j + 1] -
             smootha.data[(i + 1) * width + j - 1];

      int gxCh2 =
          com1 + com2 +
          (smootha.data[i * width + j + 1] - smootha.data[i * width + j - 1]);
      int gyCh2 = com1 - com2 +
                  (smootha.data[(i + 1) * width + j] -
                   smootha.data[(i - 1) * width + j]);

      // Prewitt for channel3
      com1 = smoothb.data[(i + 1) * width + j + 1] -
             smoothb.data[(i - 1) * width + j - 1];
      com2 = smoothb.data[(i - 1) * width + j + 1] -
             smoothb.data[(i + 1) * width + j - 1];

      int gxCh3 =
          com1 + com2 +
          (smoothb.data[i * width + j + 1] - smoothb.data[i * width + j - 1]);
      int gyCh3 = com1 - com2 +
                  (smoothb.data[(i + 1) * width + j] -
                   smoothb.data[(i - 1) * width + j]);
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

  gradImage.forEach<GradPix>([scale](GradPix &pixel, const int *pos) {
    pixel = (GradPix)(pixel * scale);
  });

  return {gradImage, dirData};
}

std::array<cv::Mat, 3> EDColor::blur(std::array<cv::Mat, 3> src,
                                     double const blurSize) {
  Mat smoothL = Mat(height, width, CV_8UC1);
  Mat smoothA = Mat(height, width, CV_8UC1);
  Mat smoothB = Mat(height, width, CV_8UC1);

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

  GaussianBlur(src[0], smoothL, kernelSize, sigma);
  GaussianBlur(src[1], smoothA, kernelSize, sigma);
  GaussianBlur(src[2], smoothB, kernelSize, sigma);

  return {smoothL, smoothA, smoothB};
}

//--------------------------------------------------------------------------------------------------------------------
// Filter edge segments using the Helmholtz principle
// Create a gradient image based on Lab encoded input image
// Then redraw the edgeImage based on the new gradient image
void EDColor::redrawEdgeImage(std::array<cv::Mat, 3> const &smoothLab) {
  cv::Mat smoothL = smoothLab[0];
  cv::Mat smoothA = smoothLab[1];
  cv::Mat smoothB = smoothLab[2];

  int maxGradValue = MAX_GRAD_VALUE;
  std::vector<double> probabilityFunctionH(maxGradValue, 0.0);

  edgeImage.setTo(0);
  cv::Mat_<GradPix> gradImage = Mat::zeros(height, width, CV_16SC1);

  std::vector<int> grads(maxGradValue, 0);

  GradPix *gradImg = gradImage.ptr<GradPix>(0);
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      // Gradient for channel1
      int com1 = smoothL.data[(i + 1) * width + j + 1] -
                 smoothL.data[(i - 1) * width + j - 1];
      int com2 = smoothL.data[(i - 1) * width + j + 1] -
                 smoothL.data[(i + 1) * width + j - 1];

      int gxCh1 = abs(
          com1 + com2 +
          (smoothL.data[i * width + j + 1] - smoothL.data[i * width + j - 1]));
      int gyCh1 = abs(com1 - com2 +
                      (smoothL.data[(i + 1) * width + j] -
                       smoothL.data[(i - 1) * width + j]));
      int ch1Grad = gxCh1 + gyCh1;

      // Gradient for channel2
      com1 = smoothA.data[(i + 1) * width + j + 1] -
             smoothA.data[(i - 1) * width + j - 1];
      com2 = smoothA.data[(i - 1) * width + j + 1] -
             smoothA.data[(i + 1) * width + j - 1];

      int gxCh2 = abs(
          com1 + com2 +
          (smoothA.data[i * width + j + 1] - smoothA.data[i * width + j - 1]));
      int gyCh2 = abs(com1 - com2 +
                      (smoothA.data[(i + 1) * width + j] -
                       smoothA.data[(i - 1) * width + j]));
      int ch2Grad = gxCh2 + gyCh2;

      // Gradient for channel3
      com1 = smoothB.data[(i + 1) * width + j + 1] -
             smoothB.data[(i - 1) * width + j - 1];
      com2 = smoothB.data[(i - 1) * width + j + 1] -
             smoothB.data[(i + 1) * width + j - 1];

      int gxCh3 = abs(
          com1 + com2 +
          (smoothB.data[i * width + j + 1] - smoothB.data[i * width + j - 1]));
      int gyCh3 = abs(com1 - com2 +
                      (smoothB.data[(i + 1) * width + j] -
                       smoothB.data[(i - 1) * width + j]));
      int ch3Grad = gxCh3 + gyCh3;

      // Take average
      int grad = (ch1Grad + ch2Grad + ch3Grad + 2) / 3;

      gradImg[i * width + j] = grad;
      grads[grad]++;
    }
  }

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
    drawFilteredSegment(segment.begin(), segment.end(), gradImage,
                        probabilityFunctionH, numberOfSegmentPieces);
  }
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
    Iterator firstPoint, Iterator lastPoint, cv::Mat_<GradPix> gradImage,
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
  uchar *edgeImg = edgeImage.ptr<uchar>(0);
  if (nfa <= EPSILON) {
    std::for_each(firstPoint, lastPoint, [&](auto const &point) {
      edgeImg[point.y * width + point.x] = 255;
    });

    return;
  }

  drawFilteredSegment(firstPoint, minGradPoint, gradImage, probabilityFunctionH,
                      numberOfSegmentPieces);
  drawFilteredSegment(std::next(minGradPoint), lastPoint, gradImage,
                      probabilityFunctionH, numberOfSegmentPieces);
}

vector<Segment> EDColor::validSegments(cv::Mat_<uchar> edgeImage,
                                       vector<Segment> segmentsIn) const {
  vector<Segment> valids;

  uchar *edgeImg = edgeImage.ptr<uchar>(0);
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
