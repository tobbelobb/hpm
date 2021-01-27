#pragma once

#include <hpm/ed/EDLines.h++>
#include <hpm/ed/EDPF.h++>

#include <utility>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wconversion"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Walloc-size-larger-than="
#endif

#define PI 3.141592653589793238462
#define TWOPI (2 * PI)

// Circular arc, circle thresholds
#define VERY_SHORT_ARC_ERROR                                                   \
  0.40 // Used for very short arcs (>= CANDIDATE_CIRCLE_RATIO1 && <
       // CANDIDATE_CIRCLE_RATIO2)
#define SHORT_ARC_ERROR                                                        \
  1.00 // Used for short arcs (>= CANDIDATE_CIRCLE_RATIO2 && <
       // HALF_CIRCLE_RATIO)
#define HALF_ARC_ERROR                                                         \
  1.25 // Used for arcs with length (>=HALF_CIRCLE_RATIO && < FULL_CIRCLE_RATIO)
#define LONG_ARC_ERROR 1.50 // Used for long arcs (>= FULL_CIRCLE_RATIO)

#define CANDIDATE_CIRCLE_RATIO1                                                \
  0.25 // 25% -- If only 25% of the circle is detected, it may be a candidate
       // for validation
#define CANDIDATE_CIRCLE_RATIO2                                                \
  0.33 // 33% -- If only 33% of the circle is detected, it may be a candidate
       // for validation
#define HALF_CIRCLE_RATIO                                                      \
  0.50 // 50% -- If 50% of a circle is detected at any point during joins, we
       // immediately make it a candidate
#define FULL_CIRCLE_RATIO                                                      \
  0.67 // 67% -- If 67% of the circle is detected, we assume that it is fully
       // covered

// Ellipse thresholds
#define CANDIDATE_ELLIPSE_RATIO                                                \
  0.50 // 50% -- If 50% of the ellipse is detected, it may be candidate for
       // validation
#define ELLIPSE_ERROR                                                          \
  1.50 // Used for ellipses. (used to be 1.65 for what reason?)

#define BOOKSTEIN 0 // method1 for ellipse fit
#define FPF 1       // method2 for ellipse fit

enum ImageStyle { NONE = 0, CIRCLES, ELLIPSES, BOTH };

// Circle equation: (x-xc)^2 + (y-yc)^2 = r^2
struct mCircle {
  cv::Point2d center;
  double r;
  mCircle(cv::Point2d _center, double _r) : center(std::move(_center)), r(_r) {}
};

// Ellipse equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
struct mEllipse {
  cv::Point2d center;
  cv::Size axes;
  double theta;
  mEllipse(cv::Point2d _center, cv::Size _axes, double _theta)
      : center(std::move(_center)), axes(std::move(_axes)), theta(_theta) {}
};

//----------------------------------------------------------
// Ellipse Equation is of the form:
// Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
//
struct EllipseEquation {
  double coeff[7]{}; // coeff[1] = A

  EllipseEquation() {
    for (double &i : coeff) {
      i = 0;
    }
  }

  auto A() -> double { return coeff[1]; }
  auto B() -> double { return coeff[2]; }
  auto C() -> double { return coeff[3]; }
  auto D() -> double { return coeff[4]; }
  auto E() -> double { return coeff[5]; }
  auto F() -> double { return coeff[6]; }
};

// ================================ CIRCLES ================================
struct Circle {
  double xc{}, yc{}, r{};  // Center (xc, yc) & radius.
  double circleFitError{}; // circle fit error
  double coverRatio{}; // Percentage of the circle covered by the arcs making up
                       // this circle [0-1]

  double *x{nullptr};
  double *y{nullptr}; // Pointers to buffers containing the pixels making up
                      // this circle
  int noPixels{};     // # of pixels making up this circle

  // If this circle is better approximated by an ellipse, we set isEllipse to
  // true & eq contains the ellipse's equation
  EllipseEquation eq;
  double ellipseFitError{}; // ellipse fit error
  bool isEllipse{};
  double majorAxisLength{}; // Length of the major axis
  double minorAxisLength{}; // Length of the minor axis
};

// ------------------------------------------- ARCS
// ----------------------------------------------------
struct MyArc {
  double xc{}, yc{}, r{};  // center x, y and radius
  double circleFitError{}; // Error during circle fit

  double sTheta{}, eTheta{}; // Start & end angle in radius
  double coverRatio{}; // Ratio of the pixels covered on the covering circle
                       // [0-1] (noPixels/circumference)

  int turn{}; // Turn direction: 1 or -1

  int segmentNo{}; // SegmentNo where this arc belongs

  int sx{}, sy{}; // Start (x, y) coordinate
  int ex{}, ey{}; // End (x, y) coordinate of the arc

  double *x{nullptr};
  double *y{nullptr}; // Pointer to buffer with the pixels making up this arc
  int noPixels{};     // # of pixels making up the arc

  bool isEllipse{};         // Did we fit an ellipse to this arc?
  EllipseEquation eq;       // If an ellipse, then the ellipse's equation
  double ellipseFitError{}; // Error during ellipse fit
};

// =============================== AngleSet ==================================

//-------------------------------------------------------------------------
// add a circular arc to the list of arcs
//
inline auto ArcLength(double sTheta, double eTheta) -> double {
  if (eTheta > sTheta) {
    return eTheta - sTheta;
  }
  { return TWOPI - sTheta + eTheta; }
}

// A fast implementation of the AngleSet class. The slow implementation is
// really bad. About 10 times slower than this!
struct AngleSetArc {
  double sTheta;
  double eTheta;
  int next; // Next AngleSetArc in the linked list
};

struct AngleSet {
  AngleSetArc angles[360]{};
  int head{};
  int next{};             // Next AngleSetArc to be allocated
  double overlapAmount{}; // Total overlap of the arcs in angleSet. Computed
                          // during set() function

  AngleSet() { clear(); }
  void clear() {
    head = -1;
    next = 0;
    overlapAmount = 0;
  }
  [[nodiscard]] auto overlapRatio() const -> double {
    return overlapAmount / (TWOPI);
  }

  void _set(double sTheta, double eTheta);
  void set(double sTheta, double eTheta);

  auto _overlap(double sTheta, double eTheta) -> double;
  auto overlap(double sTheta, double eTheta) -> double;

  void computeStartEndTheta(double &sTheta, double &eTheta);
  auto coverRatio() -> double;
};

struct EDArcs {
  MyArc *arcs;
  int noArcs;

public:
  EDArcs(int size = 10000) {
    arcs = new MyArc[size];
    noArcs = 0;
  }

  ~EDArcs() { delete arcs; }
};

//-----------------------------------------------------------------
// Buffer manager
struct BufferManager {
  double *x{nullptr};
  double *y{nullptr};
  int index;

  BufferManager(int maxSize) {
    x = new double[maxSize];
    y = new double[maxSize];
    index = 0;
  }

  ~BufferManager() {
    delete x;
    delete y;
  }

  [[nodiscard]] auto getX() const -> double * { return &x[index]; }
  [[nodiscard]] auto getY() const -> double * { return &y[index]; }
  void move(int size) { index += size; }
};

struct Info {
  int sign;     // -1 or 1: sign of the cross product
  double angle; // angle with the next line (in radians)
  bool taken;   // Is this line taken during arc detection
};

class EDCircles : public EDPF {
public:
  EDCircles(const cv::Mat &srcImage);
  EDCircles(const ED &obj);
  EDCircles(const EDColor &obj);

  auto drawResult(const cv::Mat &, ImageStyle) const -> cv::Mat;

  auto getCircles() -> std::vector<mCircle>;
  auto getEllipses() -> std::vector<mEllipse>;

  [[nodiscard]] auto getCircles() const -> std::vector<mCircle> {
    return circles;
  }
  [[nodiscard]] auto getEllipses() const -> std::vector<mEllipse> {
    return ellipses;
  }
  auto getCirclesRef() const -> std::vector<mCircle> const & { return circles; }
  auto getEllipsesRef() const -> std::vector<mEllipse> const & {
    return ellipses;
  }

  [[nodiscard]] auto getCirclesNo() const -> int;
  [[nodiscard]] auto getEllipsesNo() const -> int;

private:
  static constexpr int CIRCLE_MIN_LINE_LEN{6};
  int noEllipses;
  int noCircles;
  std::vector<mCircle> circles;
  std::vector<mEllipse> ellipses;

  Circle *circles1;
  Circle *circles2{};
  Circle *circles3{};
  int noCircles1;
  int noCircles2{};
  int noCircles3{};

  EDArcs *edarcs1;
  EDArcs *edarcs2;
  EDArcs *edarcs3;
  EDArcs *edarcs4;

  int *segmentStartLines;
  BufferManager *bm;
  Info *info;
  NFALUT *nfa{nullptr};

  void GenerateCandidateCircles();
  void DetectArcs(std::vector<LineSegment> lines);
  void ValidateCircles();
  void JoinCircles();
  void JoinArcs1();
  void JoinArcs2();
  void JoinArcs3();

  // circle utility functions
  static auto addCircle(Circle *circles, int &noCircles, double xc, double yc,
                        double r, double circleFitError, double *x, double *y,
                        int noPixels) -> Circle *;
  static auto addCircle(Circle *circles, int &noCircles, double xc, double yc,
                        double r, double circleFitError, EllipseEquation *pEq,
                        double ellipseFitError, double *x, double *y,
                        int noPixels) -> Circle *;
  static void sortCircles(Circle *circles, int noCircles);
  static auto CircleFit(const double *x, const double *y, int N, double *pxc,
                        double *pyc, double *pr, double *pe) -> bool;
  static void ComputeCirclePoints(double xc, double yc, double r, double *px,
                                  double *py, int *noPoints);
  static void sortCircle(Circle *circles, int noCircles);

  // ellipse utility functions
  static auto EllipseFit(const double *x, const double *y, int noPoints,
                         EllipseEquation *pResult, int mode = FPF) -> bool;
  static auto AllocateMatrix(int noRows, int noColumns) -> double **;
  static void A_TperB(double **A, double **B, double **_res, int _righA,
                      int _colA, int _colB);
  static void choldc(double **a, int n, double **l);
  static auto inverse(double **TB, double **InvB, int N) -> int;
  static void DeallocateMatrix(double **m, int noRows);
  static void AperB_T(double **A, double **B, double **_res, int _righA,
                      int _colA, int _colB);
  static void AperB(double **A, double **B, double **_res, int _righA,
                    int _colA, int _colB);
  static void jacobi(double **a, int n, double d[], double **v, int nrot);
  static void ROTATE(double **a, int i, int j, int k, int l, double tau,
                     double s);
  static auto computeEllipsePerimeter(EllipseEquation *eq) -> double;
  static auto ComputeEllipseError(EllipseEquation *eq, const double *px,
                                  const double *py, int noPoints) -> double;
  static auto ComputeEllipseCenterAndAxisLengths(EllipseEquation *eq,
                                                 double *pxc, double *pyc,
                                                 double *pmajorAxisLength,
                                                 double *pminorAxisLength)
      -> double;
  static void ComputeEllipsePoints(const double *pvec, double *px, double *py,
                                   int noPoints);

  // arc utility functions
  static void joinLastTwoArcs(MyArc *arcs, int &noArcs);
  static void addArc(MyArc *arcs, int &noArcs, double xc, double yc, double r,
                     double circleFitError, // Circular arc
                     double sTheta, double eTheta, int turn, int segmentNo,
                     int sx, int sy, int ex, int ey, double *x, double *y,
                     int noPixels);
  static void addArc(MyArc *arcs, int &noArcs, double xc, double yc, double r,
                     double circleFitError, // Elliptic arc
                     double sTheta, double eTheta, int turn, int segmentNo,
                     EllipseEquation *pEq, double ellipseFitError, int sx,
                     int sy, int ex, int ey, double *x, double *y, int noPixels,
                     double overlapRatio = 0.0);

  static void ComputeStartAndEndAngles(double xc, double yc, double r,
                                       const double *x, const double *y,
                                       int len, double *psTheta,
                                       double *peTheta);

  static void sortArc(MyArc *arcs, int noArcs);
};
#pragma GCC diagnostic pop
