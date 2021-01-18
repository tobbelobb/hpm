#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop

#include <hpm/ed/EDColor.h++>
#include <hpm/ed/EDTypes.h++>

enum class Direction { LEFT, RIGHT, UP, DOWN, NONE };

struct StackNode {
  int r;
  int c;         // starting pixel
  Direction dir; // direction where you are supposed to go
  int parent;    // parent chain (-1 if no parent)
};

// Used during Edge Linking
struct Chain {
  Direction dir;     // Direction of the chain
  int len;           // # of pixels in the chain
  int parent;        // Parent of this node (-1 if no parent)
  int children[2];   // Children of this node (-1 if no children)
  cv::Point *pixels; // Pointer to the beginning of the pixels array
};

enum class GradientOperator { PREWITT, SOBEL, SCHARR, LSD };

struct EdConfig {
  GradientOperator const op = GradientOperator::PREWITT;
  int const gradThresh = 20;
  int const anchorThresh = 4;
  int const scanInterval = 1;
  double const blurSize = 1.0;
  bool const sumFlag = true;
};

// Edge and segment detection from greyscale input image
class ED {

public:
  ED(cv::Mat _srcImage, EdConfig const &config);
  ED(const ED &cpyObj);
  ED(cv::Mat gradImage, std::vector<EdgeDir> dirData, int _gradThresh,
     int _anchorThresh, int _scanInterval = 1, bool selectStableAnchors = true);
  ED(EDColor const &cpyObj);

  cv::Mat getEdgeImage();
  cv::Mat getAnchorImage();
  cv::Mat getSmoothImage();
  cv::Mat getGradImage();

  int getSegmentNo();
  int getAnchorNo();

  std::vector<cv::Point> getAnchorPoints();
  std::vector<Segment> getSegments();
  std::vector<Segment> getSortedSegments();

  cv::Mat drawParticularSegments(std::vector<int> list);

protected:
  cv::Mat srcImage;
  int width;
  int height;
  uint8_t *srcImg;
  std::vector<Segment> segments;
  double blurSize;
  cv::Mat smoothImage;
  uint8_t *edgeImg;   // pointer to edge image data
  uint8_t *smoothImg; // pointer to smoothed image data
  cv::Mat edgeImage;

private:
  void ComputeGradient();
  void ComputeAnchorPoints();
  void JoinAnchorPointsUsingSortedAnchors();
  void sortAnchorsByGradValue();
  int *sortAnchorsByGradValue1();

  static int LongestChain(Chain *chains, int root);
  static int RetrieveChainNos(Chain *chains, int root, int chainNos[]);

  int anchorNos;
  std::vector<cv::Point> anchorPoints;
  std::vector<cv::Point> edgePoints;

  cv::Mat gradImage;

  std::vector<EdgeDir> dirData;

  GradientOperator op; // operation used in gradient calculation
  int gradThresh;      // gradient threshold
  int anchorThresh;    // anchor point threshold
  int scanInterval;
  bool sumFlag;

  static uint8_t constexpr ANCHOR_PIXEL{254};
  static uint8_t constexpr EDGE_PIXEL{255};
};
