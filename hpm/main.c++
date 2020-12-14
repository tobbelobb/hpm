#include <cmath>
#include <iostream>
#include <numeric>

#include <gsl/span_ext>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#endif
#include <opencv2/calib3d.hpp> // undistort
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp> // IMREAD_COLOR/IMREAD_UNCHANGED/IMREAD_GREYSCALE
#pragma GCC diagnostic pop

#include <hpm/command-line.h++>
#include <hpm/hpm.h++>
#include <hpm/individual-markers-mode.h++>
#include <hpm/solve-pnp.h++>

static auto undistort(cv::InputArray image, cv::InputArray cameraMatrix,
                      cv::InputArray distortionCoefficients) -> cv::Mat {
  cv::Mat undistortedImage =
      cv::Mat::zeros(image.rows(), image.cols(), image.type());
  cv::undistort(image, undistortedImage, cameraMatrix, distortionCoefficients);
  return undistortedImage;
}

auto main(int const argc, char **const argv) -> int {
  std::stringstream usage;
  usage << "Usage:\n"
        << *argv
        << " <camera-parameters> <marker-parameters> <image> "
           "[-h|--help] [-s|--show <value>]\n";
  CommandLine args(usage.str());

  std::string show{};
  bool showResultImage{false};
  bool showIntermediateImages{false};
  args.addArgument({"-s", "--show"}, &show,
                   "<result|intermediate|all|none>. none is the default."
                   " During any pop up you may press s to write the image, or"
                   " any other key to continue without saving.");
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
  auto *const markerParamsFileName = gsl::at(mandatoryArgs, 1);
  auto *const imageFileName = gsl::at(mandatoryArgs, 2);

  gsl::span<char *> const optionalArgs(
      &argv[MIN_ARGC], // NOLINT
      static_cast<unsigned int>(argc - MIN_ARGC));
  try {
    args.parse(optionalArgs);
  } catch (std::runtime_error const &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  showResultImage = (show == "result") or (show == "all");
  showIntermediateImages = (show == "intermediate") or (show == "all");

  if (printHelp) {
    args.printHelp();
    return 0;
  }

  auto const [cameraMatrix, distortionCoefficients, cameraWorldPose] =
      [&camParamsFileName]() -> std::tuple<cv::Mat, cv::Mat, SixDof> {
    try {
      cv::FileStorage const camParamsFile(camParamsFileName,
                                          cv::FileStorage::READ);
      if (not camParamsFile.isOpened()) {
        std::cerr << "Failed to load cam params file: " << camParamsFileName
                  << '\n';
        exit(1);
      }
      return {[&camParamsFile]() {
                cv::Mat cameraMatrix_;
                camParamsFile["camera_matrix"] >> cameraMatrix_;
                return cameraMatrix_;
              }(),
              [&camParamsFile]() {
                cv::Mat distortionCoefficients_;
                camParamsFile["distortion_coefficients"] >>
                    distortionCoefficients_;
                return distortionCoefficients_;
              }(),
              {[&camParamsFile]() {
                 Vector3d cameraRotation_;
                 camParamsFile["camera_rotation"] >> cameraRotation_;
                 return cameraRotation_;
               }(),
               [&camParamsFile]() {
                 Vector3d cameraTranslation_;
                 camParamsFile["camera_translation"] >> cameraTranslation_;
                 return cameraTranslation_;
               }()}};
    } catch (std::exception const &e) {
      std::cerr << "Could not read camera parameters from file "
                << camParamsFileName << '\n';
      std::cerr << e.what() << std::endl;
      std::exit(1);
    }
  }();

  auto const [inputMarkerPositions, markerDiameter] =
      [&markerParamsFileName]() -> std::tuple<InputMarkerPositions, double> {
    try {
      cv::FileStorage const markerParamsFile(markerParamsFileName,
                                             cv::FileStorage::READ);
      if (not markerParamsFile.isOpened()) {
        std::cerr << "Failed to load marker params file: "
                  << markerParamsFileName << '\n';
        exit(1);
      }
      return {[&markerParamsFile]() {
                InputMarkerPositions inputMarkerPositions_;
                markerParamsFile["marker_positions"] >> inputMarkerPositions_;
                return inputMarkerPositions_;
              }(),
              [&markerParamsFile]() {
                double markerDiameter_ = 0.0;
                markerParamsFile["marker_diameter"] >> markerDiameter_;
                return markerDiameter_;
              }()};
    } catch (std::exception const &e) {
      std::cerr << "Could not read marker parameters from file "
                << markerParamsFileName << '\n';
      std::cerr << e.what() << std::endl;
      std::exit(1);
    }
  }();

  const double meanFocalLength{std::midpoint(cameraMatrix.at<double>(0, 0),
                                             cameraMatrix.at<double>(1, 1))};
  PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                  cameraMatrix.at<double>(1, 2)};

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

  cv::Mat undistortedImage{
      undistort(distortedImage, cameraMatrix, distortionCoefficients)};

  DetectionResult const marks{findMarks(undistortedImage)};
  IdentifiedHpMarks const identifiedMarks{marks};

  if (identifiedMarks.allIdentified()) {
    std::optional<SixDof> const effectorPoseRelativeToCamera{
        solvePnp(cameraMatrix, inputMarkerPositions, identifiedMarks)};

    if (effectorPoseRelativeToCamera.has_value()) {
      std::cout << effectorWorldPose(effectorPoseRelativeToCamera.value(),
                                     cameraWorldPose);
    } else {
      std::cout << "Found no camera pose";
    }
  } else {
    std::cout << "Identified more or less than six markers";
  }
  std::cout << '\n';

  auto const cameraFramedPositions{findIndividualMarkerPositions(
      undistortedImage, markerDiameter, meanFocalLength, imageCenter,
      showIntermediateImages, showResultImage)};

  if (cameraFramedPositions.empty()) {
    std::cout << "No markers detected";
  }

  if (not identifiedMarks.allIdentified()) {
    std::string delimeter{};
    for (auto const &cameraFramedPosition : cameraFramedPositions) {
      // std::cout << Position{cameraFramePosition} << "mm\n";
      std::cout << delimeter << cameraFramedPosition;
      delimeter = ",\n";
    }
    std::cout << '\n';
  }

  return 0;
}
