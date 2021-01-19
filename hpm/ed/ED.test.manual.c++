#include <iostream>

#include <hpm/ed/EDLib.h++>
#include <hpm/util.h++>

using namespace cv;
using namespace std;

auto main(int argc, char **argv) -> int {
  if (argc < 2) {
    std::cout << "Usage:\n" << argv[0] << " <image> [--verbose [--show]]\n";
    return 0;
  }
  std::string const fileName{argv[1]};
  bool verbose{argc >= 3};
  bool show{argc >= 4};

  Mat greyImage{imread(fileName, IMREAD_GRAYSCALE)};
  if (show) {
    showImage(greyImage, "grey-source.png");
  }

  // Edge and segment detection from grey image
  ED ed{greyImage,
        {.op = GradientOperator::SOBEL, .gradThresh = 36, .anchorThresh = 8}};
  if (show) {
    showImage(ed.getEdgeImage(), "edges-from-grey-image.png");
  }
  if (verbose) {
    std::cout << "Segments from grey image: " << ed.getSegmentNo() << '\n';
  }

  // Line detection from grey image
  EDLines edLines{greyImage};
  if (show) {
    showImage(edLines.getLineImage(), "lines-from-grey-image.png");
  }
  if (verbose) {
    std::cout << "Lines from grey image: " << edLines.getLinesNo() << '\n';
  }

  // Line detection from segments
  // Avoids redundant detection of segments
  EDLines edLines2{ed};
  if (show) {
    showImage(edLines2.drawOnImage(), "lines-from-ed.png");
  }
  if (verbose) {
    std::cout << "Lines from segments: " << edLines2.getLinesNo() << '\n';
  }

  // Parameter free edge and segment detection from grey image
  EDPF edpf{greyImage};
  if (show) {
    showImage(edpf.getEdgeImage(), "edpf-edges-from-grey-image.png");
  }
  if (verbose) {
    std::cout << "Parameter free segments from grey image: "
              << edpf.getSegmentNo() << '\n';
  }

  // Circle and ellipse detection from grey image
  EDCircles edCirclesFromGreyImage{greyImage};
  if (show) {
    showImage(edCirclesFromGreyImage.drawResult(cv::Mat(), ImageStyle::BOTH),
              "circles-and-ellipses-from-grey-image.png");
  }
  if (verbose) {
    std::cout << "Circles from grey image: "
              << edCirclesFromGreyImage.getCirclesNo() << '\n';
  }

  // Circle and ellipse detection from (parameter free) segments
  EDCircles edCirclesFromEdpf{edpf};
  if (show) {
    showImage(edCirclesFromEdpf.drawResult(greyImage, ImageStyle::BOTH),
              "circles-and-ellipses-from-edpf.png");
  }
  if (verbose) {
    std::cout << "Circles from parameter free segments: "
              << edCirclesFromEdpf.getCirclesNo() << '\n';
  }

  // Edge and segment detection from color image
  Mat colorImage{imread(fileName, IMREAD_COLOR)};
  EDColor edColor{colorImage,
                  {.gradThresh = 36,
                   .anchorThresh = 4,
                   .blurSize = 1.5,
                   .filterSegments = true}};
  if (show) {
    showImage(edColor.getEdgeImage(), "edges-from-color-image.png");
  }
  if (verbose) {
    std::cout << "Segments from color image: " << edColor.getNumberOfSegments()
              << '\n';
  }

  // Line detection from segments from color image
  EDLines colorLines{edColor};
  if (show) {
    showImage(colorLines.getLineImage(),
              "lines-from-segments-from-color-image.png");
  }
  if (verbose) {
    std::cout << "Lines from segments from color image: "
              << colorLines.getLinesNo() << '\n';
  }

  // Circle and ellipse detection from segments from color image
  EDCircles colorCircle{edColor};
  if (verbose) {
    std::cout << "Circles from segments from color image: "
              << colorCircle.getCirclesNo() << '\n';
  }
  if (show) {
    showImage(colorCircle.drawResult(colorImage, ImageStyle::BOTH),
              "circles-and-ellipses-from-segments-from-color-image.png");
  }
  return 0;
}
