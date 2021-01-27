#pragma once

#include <hpm/ed/EDColor.h++>
#include <hpm/ed/EDTypes.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/opencv.hpp>
ENABLE_WARNINGS

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
  ED(const cv::Mat &gradImage, std::vector<EdgeDir> dirData, int _gradThresh,
     int _anchorThresh, int _scanInterval = 1, bool selectStableAnchors = true);
  ED(EDColor const &cpyObj);

  auto getEdgeImage() -> cv::Mat;
  auto getAnchorImage() -> cv::Mat;
  auto getSmoothImage() -> cv::Mat;
  auto getGradImage() -> cv::Mat;

  [[nodiscard]] auto getSegmentNo() const -> int;
  [[nodiscard]] auto getAnchorNo() const -> int;
  [[nodiscard]] auto getAnchorPoints() const -> std::vector<cv::Point>;
  [[nodiscard]] auto getSegments() const -> std::vector<Segment>;
  [[nodiscard]] auto getSortedSegments() const -> std::vector<Segment>;

  auto drawParticularSegments(std::vector<int> list) -> cv::Mat;

protected:
  cv::Mat srcImage;
  int width;
  int height;
  uint8_t *srcImg{nullptr};
  std::vector<Segment> segments;
  double blurSize{0.0};
  cv::Mat smoothImage;
  uint8_t *edgeImg{nullptr};   // pointer to edge image data
  uint8_t *smoothImg{nullptr}; // pointer to smoothed image data
  cv::Mat edgeImage;

private:
  void ComputeGradient();
  void ComputeAnchorPoints();
  void JoinAnchorPointsUsingSortedAnchors();
  void sortAnchorsByGradValue();
  auto sortAnchorsByGradValue1() -> int *;

  static auto LongestChain(Chain *chains, int root) -> int;
  static auto RetrieveChainNos(Chain *chains, int root, int chainNos[]) -> int;

  int anchorNos{};
  std::vector<cv::Point> anchorPoints;
  std::vector<cv::Point> edgePoints;

  cv::Mat gradImage;

  std::vector<EdgeDir> dirData;

  GradientOperator op; // operation used in gradient calculation
  int gradThresh{};    // gradient threshold
  int anchorThresh{};  // anchor point threshold
  int scanInterval{};
  bool sumFlag{};

  static uint8_t constexpr ANCHOR_PIXEL{254};
  static uint8_t constexpr EDGE_PIXEL{255};
};
