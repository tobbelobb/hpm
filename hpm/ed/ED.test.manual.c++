#include <iostream>

#include <hpm/ed/EDLib.h++>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  //***************************** ED Edge Segment Detection
  //***************************** Detection of edge segments from an input image
  if (argc < 2) {
    std::cout << "Usage:\n" << argv[0] << " <image> [--verbose [--show]]\n";
    return 0;
  }
  std::string const fileName{argv[1]};
  bool silent{true};
  bool show{false};
  if (argc >= 3) {
    silent = false;
    if (argc >= 4) {
      show = true;
    }
  }
  Mat testImg = imread(fileName, 0);

  if (show) {
    imshow("Source Image", testImg);
  }

  // Call ED constructor
  ED testED = ED(testImg, SOBEL_OPERATOR, 36, 8, 1, 10, 1.0,
                 true); // apply ED algorithm

  // Show resulting edge image
  Mat edgeImg = testED.getEdgeImage();
  if (show) {
    imshow("Edge Image - PRESS ANY KEY TO CONTINUE", edgeImg);
  }

  // Output number of segments
  int noSegments = testED.getSegmentNo();
  if (not silent) {
    std::cout << "Number of edge segments: " << noSegments << std::endl;
  }

  // Get edges in segment form (getSortedSegments() gives segments sorted
  // w.r.t. legnths)
  std::vector<std::vector<Point>> segments = testED.getSegments();

  //***************************** EDLINES Line Segment Detection
  //***************************** Detection of line segments from the same
  // image
  EDLines testEDLines = EDLines(testImg);
  Mat lineImg = testEDLines.getLineImage(); // draws on an empty image
  if (show) {
    imshow("Line Image 1 - PRESS ANY KEY TO CONTINUE", lineImg);
  }

  // Detection of lines segments from edge segments instead of input image
  // Therefore, redundant detection of edge segmens can be avoided
  EDLines testEDLines2{testED};
  lineImg = testEDLines2.drawOnImage(); // draws on the input image
  if (show) {
    imshow("Line Image 2  - PRESS ANY KEY TO CONTINUE", lineImg);
  }

  // Acquiring line information, i.e. start & end points
  vector<LS> lines = testEDLines2.getLines();
  int noLines = testEDLines2.getLinesNo();
  if (not silent) {
    std::cout << "Number of line segments: " << noLines << std::endl;
  }

  //************************** EDPF Parameter-free Edge Segment Detection
  //**************************
  // Detection of edge segments with parameter free ED (EDPF)

  EDPF testEDPF = EDPF(testImg);
  Mat edgePFImage = testEDPF.getEdgeImage();

  if (show) {
    imshow("Edge Image Parameter Free", edgePFImage);
  }
  if (not silent) {
    cout << "Number of edge segments found by EDPF: " << testEDPF.getSegmentNo()
         << endl;
  }
  //***************************** EDCIRCLES Circle Segment Detection
  //***************************** Detection of circles directly from the input
  // image

  EDCircles testEDCircles = EDCircles(testImg);
  Mat circleImg = testEDCircles.drawResult(false, ImageStyle::CIRCLES);

  if (show) {
    imshow("Circle Image 1", circleImg);
  }

  // Detection of circles from already available EDPF or ED image
  EDCircles testEDCircles2{testEDPF};

  // Get circle information as [cx, cy, r]
  vector<mCircle> circles = testEDCircles2.getCircles();

  // Get ellipse information as [cx, cy, a, b, theta]
  vector<mEllipse> ellipses = testEDCircles2.getEllipses();

  // Circles and ellipses will be indicated in green and red, resp.
  circleImg = testEDCircles2.drawResult(true, ImageStyle::BOTH);

  if (show) {
    imshow("CIRCLES and ELLIPSES RESULT IMAGE", circleImg);
  }

  int noCircles = testEDCircles2.getCirclesNo();
  if (not silent) {
    std::cout << "Number of circles: " << noCircles << std::endl;
  }

  //*********************** EDCOLOR Edge Segment Detection from Color Images
  //**********************

  Mat colorImg = imread(fileName);
  EDColor testEDColor{colorImg,
                      {.gradThresh = 36,
                       .anchorThresh = 4,
                       .sigma = 1.5,
                       .validateSegments = true}};
  if (show) {
    imshow("Color Edge Image - PRESS ANY KEY TO QUIT",
           testEDColor.getEdgeImage());
  }
  if (not silent) {
    cout << "Number of edge segments detected by EDColor: "
         << testEDColor.getNumberOfSegments() << endl;
  }

  // get lines from color image
  EDLines colorLine = EDLines(testEDColor);
  if (show) {
    imshow("Color Line", colorLine.getLineImage());
  }
  if (not silent) {
    std::cout << "Number of line segments: " << colorLine.getLinesNo()
              << std::endl;
  }

  // get circles from color image
  EDCircles colorCircle{testEDColor};

  if (not silent) {
    // TODO: drawResult doesnt overlay (onImage = true) when input is from
    // EDColor
    std::cout << "Number of circles: " << colorCircle.getCirclesNo()
              << std::endl;
  }
  if (show) {
    Mat circleImg2 = colorCircle.drawResult(false, ImageStyle::BOTH);
    imshow("Color Circle", circleImg2);

    waitKey();
  }

  return 0;
}
