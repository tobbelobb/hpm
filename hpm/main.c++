#include <hpm/command-line.h++>
#include <hpm/find.h++>
#include <hpm/hpm.h++>
#include <hpm/simple-types.h++>
#include <hpm/solve-pnp.h++>
#include <hpm/util.h++>

#include <gsl/span_ext>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/calib3d.hpp> // undistort
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
ENABLE_WARNINGS

#include <cmath>
#include <iostream>
#include <numeric>

using namespace hpm;

auto main(int const argc, char **const argv) -> int {
  std::stringstream usage;
  usage << "Usage:\n"
        << *argv
        << " <camera-parameters> <marker-parameters> <image> "
           "[-h|--help] [-v|--verbose] [-s|--show <value>] "
           "[-n|--no-fit-by-distance] [-c|--camera-position-calibration]\n";
  CommandLine args(usage.str());

  std::string show{};
  bool verbose{false};
  bool cameraPositionCalibration{false};
  bool showResultImage{false};
  bool showIntermediateImages{false};
  bool noFitByDistance{false};
  bool printHelp{false};
  args.addArgument({"-h", "--help"}, &printHelp, "Print this help.");
  args.addArgument({"-v", "--verbose"}, &verbose,
                   "Print rotation, translation, and reprojection_error of the "
                   "found pose. The default is to only print the translation.");
  args.addArgument({"-s", "--show"}, &show,
                   "<result|intermediate|all|none>. none is the default."
                   " During any pop up you may press s to write the image, or"
                   " any other key to continue without saving.");
  args.addArgument({"-n", "--no-fit-by-distance"}, &noFitByDistance,
                   "Don't fit the mark detection results to only those marks "
                   "who match the marks' internal distance to each other.");
  args.addArgument({"-c", "--camera-position-calibration"},
                   &cameraPositionCalibration,
                   "Output the position of the camera in a way that can be "
                   "pasted into the camera-parameters file.");

  constexpr unsigned int NUM_MANDATORY_ARGS = 3;
  constexpr unsigned int NUM_OPTIONAL_ARGS = 4;
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
  bool const fitByDistance{not noFitByDistance};

  if (printHelp) {
    args.printHelp();
    return 0;
  }

  struct Cam {
    cv::Mat const matrix;
    cv::Mat const distortion;
    hpm::SixDof const worldPose;
  };
  auto const cam = [&camParamsFileName, &cameraPositionCalibration]() -> Cam {
    cv::Mat cameraMatrix_;
    cv::Mat distortionCoefficients_;
    cv::Mat cameraRotation_(3, 1, CV_64FC1, cv::Scalar(0));
    cv::Mat cameraTranslation_(3, 1, CV_64FC1, cv::Scalar(0));
    try {
      cv::FileStorage const camParamsFile(camParamsFileName,
                                          cv::FileStorage::READ);
      if (not camParamsFile.isOpened()) {
        std::cerr << "Failed to load cam params file: " << camParamsFileName
                  << '\n';
        std::exit(1);
      }
      camParamsFile["camera_matrix"] >> cameraMatrix_;
      camParamsFile["distortion_coefficients"] >> distortionCoefficients_;
      cv::Mat cameraRotationHelper_;
      cv::Mat cameraTranslationHelper_;
      camParamsFile["camera_rotation"] >> cameraRotationHelper_;
      camParamsFile["camera_translation"] >> cameraTranslationHelper_;
      if (cameraRotationHelper_.rows == 3 and
          cameraTranslationHelper_.rows == 3) {
        cameraRotation_ = cameraRotationHelper_.clone();
        cameraTranslation_ = cameraTranslationHelper_.clone();
      } else {
        std::cout
            << "Warning! Did not find valid camera_rotation or "
               "camera_translation in "
            << camParamsFileName
            << ". Will try to calculate these based on the input image. "
               "The calculated values will only be valid if the nozzle was "
               "at the origin, and the markers were level with the print "
               "bed, when the image was taken.\n";
        cameraPositionCalibration = true;
      }
    } catch (std::exception const &e) {
      std::cerr << "Could not read camera parameters from file "
                << camParamsFileName << '\n';
      std::cerr << e.what() << std::endl;
      std::exit(1);
    }
    return {cameraMatrix_,
            distortionCoefficients_,
            {cameraRotation_, cameraTranslation_}};
  }();

  auto const [providedMarkerPositions, markerDiameter] =
      [&markerParamsFileName]() -> std::tuple<ProvidedMarkerPositions, double> {
    try {
      cv::FileStorage const markerParamsFile(markerParamsFileName,
                                             cv::FileStorage::READ);
      if (not markerParamsFile.isOpened()) {
        std::cerr << "Failed to load marker params file: "
                  << markerParamsFileName << '\n';
        exit(1);
      }
      return {[&markerParamsFile]() {
                ProvidedMarkerPositions providedMarkerPositions_;
                markerParamsFile["marker_positions"] >>
                    providedMarkerPositions_;
                return providedMarkerPositions_;
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

  const double meanFocalLength{
      std::midpoint(cam.matrix.at<double>(0, 0), cam.matrix.at<double>(1, 1))};
  PixelPosition const imageCenter{cam.matrix.at<double>(0, 2),
                                  cam.matrix.at<double>(1, 2)};

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

  cv::Mat undistortedImage(distortedImage.rows, distortedImage.cols,
                           distortedImage.type());
  cv::undistort(distortedImage, undistortedImage, cam.matrix, cam.distortion);

  auto const [points, marks] = find(
      undistortedImage, providedMarkerPositions, meanFocalLength, imageCenter,
      markerDiameter, showIntermediateImages, verbose, fitByDistance);

  if (points.allIdentified()) {
    std::optional<SixDof> const effectorPoseRelativeToCamera{
        solvePnp(cam.matrix, providedMarkerPositions, points)};

    double constexpr HIGH_REPROJECTION_ERROR{1.0};
    if (effectorPoseRelativeToCamera.has_value()) {
      if (cameraPositionCalibration) {
        if (effectorPoseRelativeToCamera.value().reprojectionError >
            HIGH_REPROJECTION_ERROR) {
          std::cout
              << "Reprojection error was too high to find good values "
                 "for camera_rotation and camera_translation. This happens "
                 "when camera_matrix, distortion_coefficients, "
                 "marker_positions, and the image don't match up well "
                 "enough.\n";
        } else {
          SixDof const camTranslation{effectorWorldPose(
              effectorPoseRelativeToCamera.value(),
              {effectorPoseRelativeToCamera.value().rotation, {0, 0, 0}})};
          std::cout << "<camera_rotation type_id=\"opencv-matrix\">\n"
                       "  <rows>3</rows>\n"
                       "  <cols>1</cols>\n"
                       "  <dt>d</dt>\n"
                       "  <data>\n    "
                    << effectorPoseRelativeToCamera.value().rotation[0] << ' '
                    << effectorPoseRelativeToCamera.value().rotation[1] << ' '
                    << effectorPoseRelativeToCamera.value().rotation[2]
                    << "\n  </data>\n"
                       "</camera_rotation>\n"
                       "<camera_translation type_id=\"opencv-matrix\">\n"
                       "  <rows>3</rows>\n"
                       "  <cols>1</cols>\n"
                       "  <dt>d</dt>\n"
                       "  <data>\n    "
                    << -camTranslation.translation[0] << ' '
                    << -camTranslation.translation[1] << ' '
                    << -camTranslation.translation[2] << "\n  </data>\n"
                    << "</camera_translation>\n";
        }
      }
      SixDof const worldPose{effectorWorldPose(
          effectorPoseRelativeToCamera.value(), cam.worldPose)};
      if (verbose) {
        std::cout << worldPose << '\n';
      }
      if (not verbose and not cameraPositionCalibration) {
        std::cout << worldPose.translation << '\n';
      }
      double constexpr HIGH_REPROJECTION_ERROR{1.0};
      if (worldPose.reprojectionError > HIGH_REPROJECTION_ERROR) {
        std::cout << "Warning! High reprojection error: "
                  << worldPose.reprojectionError << '\n';
      }
      if (showResultImage) {
        showImage(imageWith(undistortedImage, points, worldPose.translation),
                  "result.png");
      }
    } else {
      std::cout << "Found no camera pose\n";
    }
  } else {
    std::cout << "Could not identify markers\n";
  }

  auto const cameraFramedPositions{findIndividualMarkerPositions(
      marks, markerDiameter, meanFocalLength, imageCenter)};

  if (cameraFramedPositions.empty()) {
    std::cout << "No markers detected";
  }

  if (not points.allIdentified()) {
    std::string delimeter{};
    for (auto const &cameraFramedPosition : cameraFramedPositions) {
      std::cout << delimeter << cameraFramedPosition;
      delimeter = ",\n";
    }
    std::cout << '\n';
  }

  return 0;
}
