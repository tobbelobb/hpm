#include <cmath>
#include <iostream>

#include <gsl/span_ext>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>   // waitKey
#include <opencv2/imgcodecs.hpp> // IMREAD_COLOR/IMREAD_UNCHANGED/IMREAD_GREYSCALE

#include <hpm/otherfile.h++>

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
  // Configure optional command line options.
  otherfunc();
  std::stringstream usage;
  usage << "Usage:\n"
        << *argv
        << " calibration-coefficients-xml-or-json marker-diameter image\n";
  constexpr int MAX_ARGC{4};
  constexpr int MIN_ARGC{4};
  if (argc < MIN_ARGC or argc > MAX_ARGC) {
    std::cout << usage.str();
    return 0;
  }
  constexpr unsigned int NUM_MANDATORY_ARGS = 3;
  gsl::span<char *> const mandatoryArgs(&argv[1], NUM_MANDATORY_ARGS);
  auto *const camParamsFileName = gsl::at(mandatoryArgs, 0);
  double const markerDiameter = toDouble(gsl::at(mandatoryArgs, 1));
  auto *const imageFileName = gsl::at(mandatoryArgs, 2);

  cv::FileStorage const camParamsFile(camParamsFileName, cv::FileStorage::READ);
  cv::Mat const intrinsics = [&camParamsFile]() {
    cv::Mat intrinsics_;
    camParamsFile["camera_matrix"] >> intrinsics_;
    return intrinsics_;
  }();
  cv::Mat const distortion = [&camParamsFile]() {
    cv::Mat distortion_;
    camParamsFile["distortion_coefficients"] >> distortion_;
    return distortion_;
  }();

  if (markerDiameter <= 0.0) {
    std::cerr << "Need a positive marker diameter. Can not use "
              << markerDiameter << '\n';
    return 1;
  }

  cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Could not read the image: " << imageFileName << '\n';
    return 1;
  }

  // Done parsing arguments

  cv::namedWindow("Display image", cv::WINDOW_NORMAL);
  cv::resizeWindow("Display image", 300, 300);
  cv::imshow("Display image", image);
  // Wait for a keystroke in the window
  if (cv::waitKey(0) == 's') {
    cv::imwrite("image.png", image);
  }

  return 0;
}
