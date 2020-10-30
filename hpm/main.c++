#include <cmath>
#include <iostream>

#include <gsl/span_ext>

#include <opencv2/calib3d.hpp> // undistort
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>   // waitKey
#include <opencv2/imgcodecs.hpp> // IMREAD_COLOR/IMREAD_UNCHANGED/IMREAD_GREYSCALE

#include <hpm/command-line.h++>
#include <hpm/hpm.h++>

static inline auto toDouble(std::string const &s) -> double {
  std::size_t charCount = 0;
  double res = 0;
  try {
    res = std::stod(s, &charCount);
    if (s.size() != charCount) {
      throw std::invalid_argument("Could not parse all characters");
    }
  } catch (...) {
    std::cerr << "Could not parse \"" << s << "\" to a double, returning 0"
              << std::endl;
  }
  return res;
}

auto main(int const argc, char **const argv) -> int {
  std::stringstream usage;
  usage << "Usage:\n"
        << *argv
        << " <calibration-coefficients> <marker-diameter> <image> "
           "[-h|--help] [-s|--show]\n";
  CommandLine args(usage.str());

  bool showResultImage{false};
  args.addArgument({"-s", "--show"}, &showResultImage,
                   "Pops up an image of the analysis result when finished. "
                   "Useful for debugging. "
                   "Press s to write the image to image.png. "
                   "Press any other key to exit without saving the image.");
  bool printHelp{false};
  args.addArgument({"-h", "--help"}, &printHelp, "Print this help.");

  constexpr unsigned int NUM_MANDATORY_ARGS = 3;
  constexpr unsigned int NUM_OPTIONAL_ARGS = 2;
  constexpr int MAX_ARGC{NUM_MANDATORY_ARGS + NUM_OPTIONAL_ARGS + 1};
  constexpr int MIN_ARGC{NUM_MANDATORY_ARGS + 1};

  if (argc < MIN_ARGC or argc > MAX_ARGC) {
    args.printHelp();
    return 0;
  }

  gsl::span<char *> const mandatoryArgs(&argv[1], NUM_MANDATORY_ARGS); // NOLINT
  auto *const camParamsFileName = gsl::at(mandatoryArgs, 0);
  double const markerDiameter = toDouble(gsl::at(mandatoryArgs, 1));
  auto *const imageFileName = gsl::at(mandatoryArgs, 2);

  try {
    gsl::span<char *> const optionalArgs(
        &argv[MIN_ARGC], // NOLINT
        static_cast<unsigned int>(argc - MIN_ARGC));
    args.parse(optionalArgs);
  } catch (std::runtime_error const &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  if (printHelp) {
    args.printHelp();
    return 0;
  }

  cv::FileStorage const camParamsFile(camParamsFileName, cv::FileStorage::READ);
  if (not camParamsFile.isOpened()) {
    std::cerr << "Failed to load " << camParamsFileName << '\n';
    return 1;
  }
  cv::Mat const cameraMatrix = [&camParamsFile]() {
    cv::Mat cameraMatrix_;
    camParamsFile["camera_matrix"] >> cameraMatrix_;
    return cameraMatrix_;
  }();
  cv::Mat const distortionCoefficients = [&camParamsFile]() {
    cv::Mat distortionCoefficients_;
    camParamsFile["distortion_coefficients"] >> distortionCoefficients_;
    return distortionCoefficients_;
  }();
  CamParams const camParams{cameraMatrix, distortionCoefficients};

  if (markerDiameter <= 0.0) {
    std::cerr << "Need a positive marker diameter. Can not use "
              << markerDiameter << '\n';
    return 1;
  }

  cv::Mat const distortedImage = cv::imread(imageFileName, cv::IMREAD_COLOR);
  if (distortedImage.empty()) {
    std::cerr << "Could not read the image: " << imageFileName << '\n';
    return 1;
  }
  cv::Mat undistortedImage = cv::Mat::zeros(
      distortedImage.rows, distortedImage.cols, distortedImage.type());
  cv::undistort(distortedImage, undistortedImage, cameraMatrix,
                distortionCoefficients);

  std::vector<Marker> const markers{detectMarkers(camParams, undistortedImage)};

  if (markers.empty()) {
    std::cout << "No markers found\n";
  }
  for (auto const &marker : markers) {
    std::cout << marker;
  }

  if (showResultImage) {
    for (auto const &marker : markers) {
      draw(undistortedImage, marker);
    }
    constexpr auto SHOW_PIXELS_X{300};
    constexpr auto SHOW_PIXELS_Y{300};
    cv::namedWindow("Display image", cv::WINDOW_NORMAL);
    cv::resizeWindow("Display image", SHOW_PIXELS_X, SHOW_PIXELS_Y);
    cv::imshow("Display image", undistortedImage);
    // Wait for a keystroke in the window
    if (cv::waitKey(0) == 's') {
      cv::imwrite("image.png", undistortedImage);
    }
  }

  return 0;
}
