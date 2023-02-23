#include <hpm/command-line.h++>
#include <hpm/ellipse-detector.h++>
#include <hpm/find.h++>
#include <hpm/hpm.h++>
#include <hpm/marks.h++>
#include <hpm/simple-types.h++>
#include <hpm/solve-pnp.h++>
#include <hpm/util.h++>

#include <gsl/span_ext>

#include <hpm/warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
ENABLE_WARNINGS

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

using namespace hpm;

struct Cam {
  cv::Mat const matrix;
  cv::Mat const distortion;
  hpm::SixDof const worldPose;
  bool calibrationNeeded = false;
};
static auto getCamParams(std::string const &camParamsFileName) -> Cam {
  cv::Mat cameraMatrix_;
  cv::Mat distortionCoefficients_;
  cv::Mat cameraRotation_(3, 1, CV_64FC1, cv::Scalar(0));
  cv::Mat cameraTranslation_(3, 1, CV_64FC1, cv::Scalar(0));
  bool calibrationNeeded = false;
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
      std::cout << "Warning! Did not find valid camera_rotation or "
                   "camera_translation in "
                << camParamsFileName
                << ". Will try to calculate these based on the input image. "
                   "The calculated values will only be valid if the nozzle was "
                   "at the origin, and the markers were level with the print "
                   "bed, when the image was taken.\n";
      calibrationNeeded = true;
    }
  } catch (std::exception const &e) {
    std::cerr << "Could not read camera parameters from file "
              << camParamsFileName << '\n';
    std::cerr << e.what() << std::endl;
    std::exit(1);
  }
  return {cameraMatrix_,
          distortionCoefficients_,
          {cameraRotation_, cameraTranslation_},
          calibrationNeeded};
}

static auto getMarkerParams(std::string const &fileName,
                            std::string const &whichMarkers) -> MarkerParams {
  try {
    cv::FileStorage const markerParamsFile(fileName, cv::FileStorage::READ);
    if (not markerParamsFile.isOpened()) {
      std::cerr << "Failed to load marker params file: " << fileName << '\n';
      exit(1);
    }
    return {
        [&]() {
          ProvidedMarkerPositions effectorMarkerPositions_;
          markerParamsFile[whichMarkers] >> effectorMarkerPositions_;
          return effectorMarkerPositions_;
        }(),
        [&]() {
          double markerDiameter_ = 0.0;
          markerParamsFile[whichMarkers]["marker_diameter"] >> markerDiameter_;
          return markerDiameter_;
        }(),
        [&]() {
          std::string markerType_;
          markerParamsFile[whichMarkers]["marker_type"] >> markerType_;
          std::transform(std::begin(markerType_), std::end(markerType_),
                         std::begin(markerType_),
                         [](unsigned char c) { return std::tolower(c); });
          if (markerType_ == "disk" or markerType_ == "disc" or
              markerType_ == "disks" or markerType_ == "discs") {
            return MarkerType::DISK;
          }
          return MarkerType::SPHERE;
        }(),
    };
  } catch (std::exception const &e) {
    std::cerr << "Could not read marker parameters from file " << fileName
              << '\n';
    std::cerr << e.what() << std::endl;
    std::exit(1);
  }
}

auto main(int const argc, char **const argv) -> int {
  std::stringstream usage;
  usage << "Usage:\n"
        << *argv
        << " <camera-parameters> <marker-parameters> <image> "
           "[-h|--help] [-v|--verbose] [-s|--show <value>] "
           "[-n|--no-fit-by-distance] [-c|--camera-position-calibration] "
           "[-t|--try-hard] [-f|--force-try-hard] [-b|--bed-reference] "
           "[-r|--reprojection-error-limit <value>]\n";
  CommandLine args(usage.str());

  std::string show{};
  bool verbose{false};
  bool cameraPositionCalibration{false};
  bool showResultImage{false};
  bool showIntermediateImages{false};
  bool noFitByDistance{false};
  bool printHelp{false};
  bool tryHard{false};
  bool forceTryHard{false};
  bool useBedReference{false};
  double reprojectionErrorLimit{1.0};
  args.addArgument({"-h", "--help"}, &printHelp, "Print this help.");
  args.addArgument({"-v", "--verbose"}, &verbose,
                   "Print rotation, translation, and reprojection_error of the "
                   "found pose. The default is to only print the translation.");
  args.addArgument(
      {"-s", "--show"}, &show,
      "<result|r|intermediate|i|all|a|none|n>. none is the default. During any "
      "pop up you may press s to write the image, or q to stop showing images, "
      "or any other key to continue.");
  args.addArgument({"-n", "--no-fit-by-distance"}, &noFitByDistance,
                   "Don't fit the mark detection results to only those marks "
                   "who match the marks' internal distance to each other.");
  args.addArgument({"-c", "--camera-position-calibration"},
                   &cameraPositionCalibration,
                   "Output the position of the camera in a way that can be "
                   "pasted into the camera-parameters file.");
  args.addArgument(
      {"-t", "--try-hard"}, &tryHard,
      "Try harder (but slower) to find a good position. If one marker was "
      "slightly mis-detected, this option will make the program find decent "
      "values based on the other markers, and ignore the mis-detected one.");
  args.addArgument({"-f", "--force-try-hard"}, &forceTryHard,
                   "Works like --try-hard, but will always use only the best 5 "
                   "markers for position calculation, even if no markers were "
                   "mis-detected. Setting both --try-hard and --force-try-hard "
                   "has the same effect as only setting --force-try-hard.");
  args.addArgument(
      {"-b", "--bed-reference"}, &useBedReference,
      "If you have a set of markers on a flat background surface (\"the "
      "bed\"), and have specified bed marker positions, type, and diameter in "
      "the marker-parameters file, then the bed reference option will make hpm "
      "calculate effector pose relative to the bed instead of relative to the "
      "camera. This can give more stable results if the camera isn't "
      "completely still between measurements, and makes measurements far less "
      "sensitive to having correct camera_translation and camera_rotation "
      "values in the camera-parameters file.");
  args.addArgument(
      {"-r", "--reprojection-error-limit"}, &reprojectionErrorLimit,
      "<number>. Default is 1.0. A lower number rejects more unceirtain "
      "results. A higher number accepts more unceirtain results.");

  constexpr unsigned int NUM_MANDATORY_ARGS = 3;
  constexpr unsigned int NUM_OPTIONAL_ARGS = 9;
  constexpr unsigned int NUM_OPTIONAL_VALS = 2;
  constexpr int MAX_ARGC{NUM_MANDATORY_ARGS + NUM_OPTIONAL_ARGS +
                         NUM_OPTIONAL_VALS + 1};
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
  {
    bool const showAll{(show == "all") or (show == "a")};
    showResultImage = (show == "result") or (show == "r") or showAll;
    showIntermediateImages =
        (show == "intermediate") or (show == "i") or showAll;
  }
  bool const fitByDistance{not noFitByDistance};
  FinderConfig const finderConfig{showIntermediateImages, verbose,
                                  fitByDistance};

  tryHard = tryHard or forceTryHard;

  if (printHelp) {
    args.printHelp();
    return 0;
  }

  auto const cam{getCamParams(camParamsFileName)};
  cameraPositionCalibration =
      cameraPositionCalibration or cam.calibrationNeeded;

  auto const effectorMarkerParams{
      getMarkerParams(markerParamsFileName, "effector_markers")};

  auto const bedMarkerParams = [&markerParamsFileName,
                                useBedReference]() -> MarkerParams {
    if (useBedReference) {
      return getMarkerParams(markerParamsFileName, "bed_markers");
    }
    return {};
  }();

  const double meanFocalLength{
      std::midpoint(cam.matrix.at<double>(0, 0), cam.matrix.at<double>(1, 1))};
  PixelPosition const imageCenter{cam.matrix.at<double>(0, 2),
                                  cam.matrix.at<double>(1, 2)};

  if (effectorMarkerParams.m_diameter <= 0.0) {
    std::cerr << "Need a positive effector marker diameter. Can not use "
              << effectorMarkerParams.m_diameter << '\n';
    return 1;
  }
  if (useBedReference and bedMarkerParams.m_diameter <= 0.0) {
    std::cerr << "Need a positive bed marker diameter. Can not use "
              << bedMarkerParams.m_diameter << '\n';
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
  FinderImage const finderImage{undistortedImage, meanFocalLength, imageCenter};

  CameraFramedPosition expectedNormalDirection{0.0, 0.0, 0.0};
  if (effectorMarkerParams.m_type == MarkerType::DISK) {
    cv::Matx33d rotationMatrix;
    cv::Rodrigues(cam.worldPose.rotation, rotationMatrix);
    expectedNormalDirection =
        CameraFramedPosition{rotationMatrix * hpm::Vector3d{0.0, 0.0, 1.0}};
  }

  std::vector<hpm::Ellipse> ellipses{
      ellipseDetect(undistortedImage, showIntermediateImages)};

  auto const effectorMarks{
      findMarks(finderImage, ellipses, effectorMarkerParams, finderConfig,
                expectedNormalDirection, tryHard, "effector-")};
  auto const bedMarks = [&]() -> std::vector<hpm::Ellipse> {
    if (useBedReference) {
      ellipses.erase(std::remove_if(std::begin(ellipses), std::end(ellipses),
                                    [&](auto x) {
                                      return find(std::begin(effectorMarks),
                                                  std::end(effectorMarks),
                                                  x) != end(effectorMarks);
                                    }),
                     std::end(ellipses));
      return findMarks(finderImage, ellipses, bedMarkerParams, finderConfig,
                       expectedNormalDirection, tryHard, "bed-");
    }
    return {};
  }();

  SolvePnpPoints effectorPoints{effectorMarks,
                                effectorMarkerParams.m_diameter,
                                finderImage.m_focalLength,
                                finderImage.m_center,
                                effectorMarkerParams.m_type,
                                expectedNormalDirection};

  bool const allEffectorPointsWereIdentified{effectorPoints.allIdentified()};

  if (allEffectorPointsWereIdentified) {
    std::optional<SixDof> effectorPoseRelativeToCamera{};
    if (not forceTryHard) {
      effectorPoseRelativeToCamera =
          solvePnp(cam.matrix, effectorMarkerParams.m_providedMarkerPositions,
                   effectorPoints);
    }
    if (forceTryHard or
        (tryHard and (not effectorPoseRelativeToCamera.has_value() or
                      effectorPoseRelativeToCamera.value().reprojectionError >
                          reprojectionErrorLimit))) {
      effectorPoints =
          SolvePnpPoints(effectorMarks, effectorMarkerParams.m_diameter,
                         finderImage.m_focalLength, finderImage.m_center,
                         effectorMarkerParams.m_type, expectedNormalDirection);
      effectorPoseRelativeToCamera = tryHardSolvePnp(
          cam.matrix, effectorMarkerParams.m_providedMarkerPositions,
          effectorPoints);
    }

    SolvePnpPoints bedPoints{bedMarks,
                             bedMarkerParams.m_diameter,
                             finderImage.m_focalLength,
                             finderImage.m_center,
                             bedMarkerParams.m_type,
                             expectedNormalDirection};
    std::optional<SixDof> bedPoseRelativeToCamera{};
    if (useBedReference and not forceTryHard) {
      if (bedPoints.allIdentified()) {
        bedPoseRelativeToCamera = solvePnp(
            cam.matrix, bedMarkerParams.m_providedMarkerPositions, bedPoints);
      }
    }
    if (useBedReference and
        (forceTryHard or
         (tryHard and (not bedPoseRelativeToCamera.has_value() or
                       bedPoseRelativeToCamera.value().reprojectionError >
                           reprojectionErrorLimit)))) {
      bedPoints =
          SolvePnpPoints(bedMarks, bedMarkerParams.m_diameter,
                         finderImage.m_focalLength, finderImage.m_center,
                         bedMarkerParams.m_type, expectedNormalDirection);
      bedPoseRelativeToCamera = tryHardSolvePnp(
          cam.matrix, bedMarkerParams.m_providedMarkerPositions, bedPoints);
    }

    if (effectorPoseRelativeToCamera.has_value()) {
      if (cameraPositionCalibration) {
        try {
          if (effectorPoseRelativeToCamera.value().reprojectionError >
              reprojectionErrorLimit) {
            if ((not bedPoseRelativeToCamera.has_value()) or
                bedPoseRelativeToCamera.value().reprojectionError >
                    reprojectionErrorLimit) {
              std::cout
                  << "Error: Effector reprojection error was "
                  << effectorPoseRelativeToCamera.value().reprojectionError
                  << ", and bed reprojection error was "
                  << (bedPoseRelativeToCamera.has_value()
                          ? bedPoseRelativeToCamera.value().reprojectionError
                          : std::nan("undefined"))
                  << ". That is worse than the limit " << reprojectionErrorLimit
                  << ". Therefore we can't find good camera_rotation and "
                     "camera_translation values. A likely cause for the high "
                     "reprojection error is that the configured camera_matrix, "
                     "distortion_coefficients, and/or marker positions don't "
                     "match up well enough with what's found on the image. "
                     "Another cause might be that the marker detector "
                     "algorithm "
                     "makes a mistake. Try re-running with '--show all' to "
                     "verify if this is the case.";
            } else {
              SixDof const camTranslation{
                  effectorWorldPose( // Treat bed as an effector
                      bedPoseRelativeToCamera.value(),
                      {bedPoseRelativeToCamera.value().rotation, {0, 0, 0}})};
              std::cout << "<camera_rotation type_id=\"opencv-matrix\">\n"
                           "  <!-- Based on bed pose -->\n"
                           "  <rows>3</rows>\n"
                           "  <cols>1</cols>\n"
                           "  <dt>d</dt>\n"
                           "  <data>\n    "
                        << bedPoseRelativeToCamera.value().rotation[0] << ' '
                        << bedPoseRelativeToCamera.value().rotation[1] << ' '
                        << bedPoseRelativeToCamera.value().rotation[2]
                        << "\n  </data>\n"
                           "</camera_rotation>\n"
                           "<camera_translation type_id=\"opencv-matrix\">\n"
                           "  <!-- Based on bed pose -->\n"
                           "  <rows>3</rows>\n"
                           "  <cols>1</cols>\n"
                           "  <dt>d</dt>\n"
                           "  <data>\n    "
                        << -camTranslation.translation[0] << ' '
                        << -camTranslation.translation[1] << ' '
                        << -camTranslation.translation[2] << "\n  </data>\n"
                        << "</camera_translation>";
            }
          } else {
            SixDof const camTranslation{effectorWorldPose(
                effectorPoseRelativeToCamera.value(),
                {effectorPoseRelativeToCamera.value().rotation, {0, 0, 0}})};

            std::cout << "<camera_rotation type_id=\"opencv-matrix\">\n"
                         "  <!-- Based on effector pose -->\n"
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
                         "  <!-- Based on effector pose -->\n"
                         "  <rows>3</rows>\n"
                         "  <cols>1</cols>\n"
                         "  <dt>d</dt>\n"
                         "  <data>\n    "
                      << -camTranslation.translation[0] << ' '
                      << -camTranslation.translation[1] << ' '
                      << -camTranslation.translation[2] << "\n  </data>\n"
                      << "</camera_translation>";

              std::ofstream file ("myCamExtParams.xml");

          	if (file.is_open()) {
          		  file << "<?xml version='1.0'?>\n";
			          file << "<opencv_storage>\n";
          		  file << "<camera_rotation type_id=\"opencv-matrix\">\n"
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
                      	file << "</opencv_storage>";
          		  file.close();
          	} else {
          		  std::cout << "Unable to open file";
          	}
          }
        } catch (std::exception const &e) {
          std::cerr << "Catch\n";
          std::cerr << e.what() << std::endl;
          std::exit(1);
        }
      }
      auto const worldPose = [&]() -> SixDof {
        if (useBedReference and bedPoseRelativeToCamera.has_value()) {
          return effectorPoseRelativeToBed(effectorPoseRelativeToCamera.value(),
                                           bedPoseRelativeToCamera.value());
        }
        if (useBedReference and (not bedPoseRelativeToCamera.has_value())) {
          std::cout << "Warning: Found no bed pose. Calculating effector pose "
                       "based on camera pose.\n";
        }
        return effectorWorldPose(effectorPoseRelativeToCamera.value(),
                                 cam.worldPose);
      }();
      if (verbose) {
        if (cameraPositionCalibration) {
          std::cout << '\n';
        }
        std::cout << worldPose;
        if (tryHard) {
          std::cout << "\nUsed "
                    << std::count_if(std::begin(effectorPoints.m_identified),
                                     std::end(effectorPoints.m_identified),
                                     std::identity())
                    << " identified effector markers";
          if (useBedReference and bedPoseRelativeToCamera.has_value()) {
            std::cout << "\nUsed "
                      << std::count_if(std::begin(bedPoints.m_identified),
                                       std::end(bedPoints.m_identified),
                                       std::identity())
                      << " identified bed markers";
          }
        }
        if (useBedReference and effectorPoseRelativeToCamera.has_value()) {
          std::cout << "\nEffector reprojection error: "
                    << effectorPoseRelativeToCamera.value().reprojectionError;
          if (bedPoseRelativeToCamera.has_value()) {
            std::cout << "\nBed reprojection error: "
                      << bedPoseRelativeToCamera.value().reprojectionError;
          }
        }
      }
      if (not verbose and not cameraPositionCalibration) {
        std::cout << worldPose.translation << ";";
      }
      if (not(cameraPositionCalibration) and
          worldPose.reprojectionError > reprojectionErrorLimit) {
        std::cout << " Warning! High reprojection error: "
                  << worldPose.reprojectionError;
      }
      std::cout << std::endl;
      if (showResultImage) {
        if (useBedReference) {
          showImage(imageWith(undistortedImage, effectorPoints, bedPoints,
                              worldPose.translation),
                    "result_w_bed.png");
        } else {
          showImage(imageWith(undistortedImage, effectorPoints,
                              worldPose.translation),
                    "result.png");
        }
      }
    } else {
      std::cout << "Found no camera pose\n";
    }
  } else {
    std::cout << "Could not identify effector markers" << std::endl;
    if (showResultImage) {
      showImage(undistortedImage, "result.png");
    }
  }

  if (not(allEffectorPointsWereIdentified) and verbose) {
    auto const cameraFramedPositions{findIndividualMarkerPositions(
        effectorMarks, effectorMarkerParams.m_diameter, meanFocalLength,
        imageCenter, effectorMarkerParams.m_type, expectedNormalDirection)};
    if (cameraFramedPositions.empty()) {
      std::cout << "No markers detected\n";
    }
    std::string delimeter{};
    for (auto const &cameraFramedPosition : cameraFramedPositions) {
      std::cout << delimeter << cameraFramedPosition;
      delimeter = ",\n";
    }
    std::cout << '\n';
  }

  return 0;
}
