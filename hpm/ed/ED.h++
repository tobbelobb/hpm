#pragma once

#include <opencv2/opencv.hpp>

#include <hpm/ed/EDColor.h++>
#include <hpm/ed/EDTypes.h++>

#define ANCHOR_PIXEL 254
#define EDGE_PIXEL 255

#define LEFT 1
#define RIGHT 2
#define UP 3
#define DOWN 4

enum GradientOperator {
  PREWITT_OPERATOR = 101,
  SOBEL_OPERATOR = 102,
  SCHARR_OPERATOR = 103,
  LSD_OPERATOR = 104
};

struct StackNode {
  int r, c;   // starting pixel
  int parent; // parent chain (-1 if no parent)
  int dir;    // direction where you are supposed to go
};

// Used during Edge Linking
struct Chain {

  int dir;           // Direction of the chain
  int len;           // # of pixels in the chain
  int parent;        // Parent of this node (-1 if no parent)
  int children[2];   // Children of this node (-1 if no children)
  cv::Point *pixels; // Pointer to the beginning of the pixels array
};

class ED {

public:
  ED(cv::Mat _srcImage, GradientOperator _op = PREWITT_OPERATOR,
     int _gradThresh = 20, int _anchorThresh = 0, int _scanInterval = 1,
     int _minPathLen = 10, double _sigma = 1.0, bool _sumFlag = true);
  ED(const ED &cpyObj);
  ED(cv::Mat gradImage, std::vector<EdgeDir> dirData, int _gradThresh,
     int _anchorThresh, int _scanInterval = 1, int _minPathLen = 10,
     bool selectStableAnchors = true);
  ED(EDColor &cpyObj);

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
  uchar *srcImg;
  std::vector<std::vector<cv::Point>> segments;
  double sigma; // Gaussian sigma
  cv::Mat smoothImage;
  uchar *edgeImg;   // pointer to edge image data
  uchar *smoothImg; // pointer to smoothed image data
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
};
