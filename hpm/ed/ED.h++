#pragma once

#include <opencv2/opencv.hpp>

#include <hpm/ed/EDColor.h++>
#include <hpm/ed/EDTypes.h++>

enum class Direction { LEFT, RIGHT, UP, DOWN, NONE };
struct StackNode {
  int r, c;      // starting pixel
  int parent;    // parent chain (-1 if no parent)
  Direction dir; // direction where you are supposed to go
};
// Used during Edge Linking
struct Chain {
  Direction dir;     // Direction of the chain
  int len;           // # of pixels in the chain
  int parent;        // Parent of this node (-1 if no parent)
  int children[2];   // Children of this node (-1 if no children)
  cv::Point *pixels; // Pointer to the beginning of the pixels array
};

enum GradientOperator {
  PREWITT_OPERATOR = 101,
  SOBEL_OPERATOR = 102,
  SCHARR_OPERATOR = 103,
  LSD_OPERATOR = 104
};

// Edge and segment detection from greyscale input image
class ED {

public:
  ED(cv::Mat _srcImage, GradientOperator _op = PREWITT_OPERATOR,
     int _gradThresh = 20, int _anchorThresh = 0, int _scanInterval = 1,
     int _minPathLen = 10, double _sigma = 1.0, bool _sumFlag = true);
  ED(const ED &cpyObj);
  ED(cv::Mat gradImage, std::vector<EdgeDir> dirData, int _gradThresh,
     int _anchorThresh, int _scanInterval = 1, int _minPathLen = 10,
     bool selectStableAnchors = true);
  ED(EDColor const &cpyObj);

  cv::Mat getEdgeImage();
  cv::Mat getAnchorImage();
  cv::Mat getSmoothImage();
  cv::Mat getGradImage();

  int getSegmentNo();
  int getAnchorNo();

  std::vector<cv::Point> getAnchorPoints();
  std::vector<std::vector<cv::Point>> getSegments();
  std::vector<std::vector<cv::Point>> getSortedSegments();

  cv::Mat drawParticularSegments(std::vector<int> list);

protected:
  int width;  // width of source image
  int height; // height of source image
  uint8_t *srcImg;
  std::vector<std::vector<cv::Point>> segments;
  double sigma; // Gaussian sigma
  cv::Mat smoothImage;
  uint8_t *edgeImg;   // pointer to edge image data
  uint8_t *smoothImg; // pointer to smoothed image data
  int minPathLen;
  cv::Mat srcImage;

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

  cv::Mat edgeImage;
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
