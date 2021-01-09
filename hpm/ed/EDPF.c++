#include <hpm/ed/EDPF.h++>

using namespace cv;
using namespace std;

EDPF::EDPF(Mat srcImage) : ED(srcImage, PREWITT_OPERATOR, 11, 3) {
  // Validate Edge Segments
  sigma /= 2.5;
  GaussianBlur(srcImage, smoothImage, Size(),
               sigma); // calculate kernel from sigma

  validateEdgeSegments();
}

EDPF::EDPF(ED obj) : ED(obj) {
  // Validate Edge Segments
  sigma /= 2.5;
  GaussianBlur(srcImage, smoothImage, Size(),
               sigma); // calculate kernel from sigma

  validateEdgeSegments();
}

EDPF::EDPF(EDColor obj) : ED(obj) {}

void EDPF::validateEdgeSegments() {

  memset(edgeImg, 0, width * height); // clear edge image

  H = new double[MAX_GRAD_VALUE];
  memset(H, 0, sizeof(double) * MAX_GRAD_VALUE);

  gradImg = ComputePrewitt3x3();

  // Compute np: # of segment pieces
  // Does this underestimate the number of pieces of edge segments?
  // What's the correct value?
  np = 0;
  for (size_t i = 0; i < segmentNos; ++i) {
    size_t const len = segmentPoints[i].size();
    np += (len * (len - 1)) / 2;
  }

  // Validate segments
  for (size_t i = 0; i < segmentNos; i++) {
    TestSegment(i, 0, segmentPoints[i].size() - 1);
  }

  ExtractNewSegments();

  // clean space
  delete[] H;
  delete[] gradImg;
}

short *EDPF::ComputePrewitt3x3() {
  short *gradImg = new short[width * height];
  memset(gradImg, 0, sizeof(short) * width * height);

  int *grads = new int[MAX_GRAD_VALUE];
  memset(grads, 0, sizeof(int) * MAX_GRAD_VALUE);

  for (size_t i = 1; i < height - 1; i++) {
    for (size_t j = 1; j < width - 1; j++) {
      // Prewitt Operator in horizontal and vertical direction
      // A B C
      // D x E
      // F G H
      // gx = (C-A) + (E-D) + (H-F)
      // gy = (F-A) + (G-B) + (H-C)
      //
      // To make this faster:
      // com1 = (H-A)
      // com2 = (C-F)
      // Then: gx = com1 + com2 + (E-D) = (H-A) + (C-F) + (E-D) = (C-A) + (E-D)
      // + (H-F)
      //       gy = com1 - com2 + (G-B) = (H-A) - (C-F) + (G-B) = (F-A) + (G-B)
      //       + (H-C)
      //
      int com1 =
          srcImg[(i + 1) * width + j + 1] - srcImg[(i - 1) * width + j - 1];
      int com2 =
          srcImg[(i - 1) * width + j + 1] - srcImg[(i + 1) * width + j - 1];

      int gx = abs(com1 + com2 +
                   (srcImg[i * width + j + 1] - srcImg[i * width + j - 1]));
      int gy = abs(com1 - com2 +
                   (srcImg[(i + 1) * width + j] - srcImg[(i - 1) * width + j]));

      int g = gx + gy;

      gradImg[i * width + j] = static_cast<short>(g);
      grads[g]++;
    }
  }

  // Compute probability function H
  int size = (width - 2) * (height - 2);

  for (int i = MAX_GRAD_VALUE - 1; i > 0; i--)
    grads[i - 1] += grads[i];

  for (int i = 0; i < MAX_GRAD_VALUE; i++)
    H[i] = (double)grads[i] / ((double)size);

  delete[] grads;
  return gradImg;
}

//----------------------------------------------------------------------------------
// Resursive validation using half of the pixels as suggested by DMM algorithm
// We take pixels at Nyquist distance, i.e., 2 (as suggested by DMM)
//
void EDPF::TestSegment(size_t i, size_t index1, size_t index2) {

  int chainLen = index2 - index1 + 1;
  if (chainLen < minPathLen) {
    return;
  }

  // Test from index1 to index2. If OK, then we are done. Otherwise, split into
  // two and recursively test the left & right halves

  // First find the min. gradient along the segment
  int minGrad = 1 << 30;
  int minGradIndex;
  for (size_t k = index1; k <= index2; k++) {
    size_t r = segmentPoints[i][k].y;
    size_t c = segmentPoints[i][k].x;
    if (gradImg[r * width + c] < minGrad) {
      minGrad = gradImg[r * width + c];
      minGradIndex = k;
    }
  }

  // Compute nfa
  double constexpr DIV_FOR_TEST_SEGMENT{2.25};
  double nfa = NFA(H[minGrad], (int)(chainLen / DIV_FOR_TEST_SEGMENT));

  if (nfa <= EPSILON) {
    for (size_t k = index1; k <= index2; k++) {
      size_t r = segmentPoints[i][k].y;
      size_t c = segmentPoints[i][k].x;

      edgeImg[r * width + c] = 255;
    }

    return;
  }

  // Split into two halves. We divide at the point where the gradient is the
  // minimum
  size_t end = minGradIndex - 1;
  while (end > index1) {
    int r = segmentPoints[i][end].y;
    int c = segmentPoints[i][end].x;

    // segfaults
    if (gradImg[r * width + c] <= minGrad) {
      end--;
    } else {
      break;
    }
  }

  size_t start = minGradIndex + 1;
  while (start < index2) {
    size_t r = segmentPoints[i][start].y;
    size_t c = segmentPoints[i][start].x;

    if (gradImg[r * width + c] <= minGrad) {
      start++;
    } else {
      break;
    }
  }

  TestSegment(i, index1, end);
  TestSegment(i, start, index2);
}

//----------------------------------------------------------------------------------------------
// After the validation of the edge segments, extracts the valid ones
// In other words, updates the valid segments' pixel arrays and their lengths
//
void EDPF::ExtractNewSegments() {
  vector<vector<Point>> validSegments;
  int noSegments = 0;

  for (size_t i = 0; i < segmentNos; i++) {
    int start = 0;
    while (start < segmentPoints[i].size()) {

      while (start < segmentPoints[i].size()) {
        int r = segmentPoints[i][start].y;
        int c = segmentPoints[i][start].x;

        if (edgeImg[r * width + c])
          break;
        start++;
      }

      int end = start + 1;
      while (end < segmentPoints[i].size()) {
        int r = segmentPoints[i][end].y;
        int c = segmentPoints[i][end].x;

        if (edgeImg[r * width + c] == 0)
          break;
        end++;
      }

      int len = end - start;
      if (len >= 10) {
        // A new segment. Accepted only only long enough (whatever that means)
        // segments[noSegments].pixels = &map->segments[i].pixels[start];
        // segments[noSegments].noPixels = len;
        validSegments.push_back(vector<Point>());
        vector<Point> subVec(&segmentPoints[i][start],
                             &segmentPoints[i][end - 1]);
        validSegments[noSegments] = subVec;
        noSegments++;
      }

      start = end + 1;
    }
  }

  // Copy to ed
  segmentPoints = validSegments;

  segmentNos = noSegments;
}

//---------------------------------------------------------------------------
// Number of false alarms code as suggested by Desolneux, Moisan and Morel (DMM)
//
double EDPF::NFA(double prob, int len) {
  double nfa = np;
  for (int i = 0; i < len && nfa > EPSILON; i++) {
    nfa *= prob;
  }

  return nfa;
}
