#pragma once

#include <vector>

#include <hpm/ed/EDTypes.h++>

double NFA(double prob, int len, int const numberOfSegmentPieces);

std::vector<Segment> validSegments(cv::Mat edgeImageIn,
                                   std::vector<Segment> const &segmentsIn);

//----------------------------------------------------------------------------------
// Resursive validation using half of the pixels as suggested by DMM algorithm
// We take pixels at Nyquist distance, i.e., 2 (as suggested by DMM)
template <typename Iterator>
void drawFilteredSegment(Iterator firstPoint, Iterator lastPoint,
                         cv::Mat edgeImageIn, cv::Mat_<GradPix> gradImage,
                         std::vector<double> const &probabilityFunctionH,
                         int const numberOfSegmentPieces) {
  int const width{gradImage.cols};

  int const chainLen = std::distance(firstPoint, lastPoint);
  if (chainLen < MIN_SEGMENT_LEN) {
    return;
  }

  // First find the min. gradient along the segment
  GradPix const *gradImg = gradImage.ptr<GradPix>(0);
  auto minGradPoint = std::min_element(
      firstPoint, lastPoint, [&](cv::Point const &p0, cv::Point const &p1) {
        return gradImg[p0.y * width + p0.x] < gradImg[p1.y * width + p1.x];
      });
  int minGrad = gradImg[(*minGradPoint).y * width + (*minGradPoint).x];

  double nfa = NFA(probabilityFunctionH[minGrad],
                   static_cast<int>(chainLen / 2.25), numberOfSegmentPieces);

  // Draw subsegment on edgeImage
  uint8_t *edgeImg = edgeImageIn.ptr<uint8_t>(0);
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
