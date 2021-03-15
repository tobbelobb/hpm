#include <hpm/ellipse-detector.h++>
#include <hpm/marks.h++>
#include <hpm/test-util.h++>
#include <hpm/util.h++>

#include <boost/ut.hpp> //import boost.ut;

auto main() -> int {
  using namespace hpm;
  using namespace boost::ut;

  auto constexpr EPS2{0.01_d};
  auto constexpr EPS3{0.001_d};
  auto constexpr EPS4{0.0001_d};
  auto constexpr EPS9{0.000000001_d};
  auto constexpr EPS11{0.00000000001_d};

  skip / "identify sphere-marks according to provided positions"_test = [&] {
    // clang-format off
    cv::Mat const cameraMatrix = (cv::Mat_<double>(3, 3) << 3000.0,    0.0,
    1000.0,
                                                               0.0, 3000.0,
                                                               1000.0, 0.0,
                                                               0.0,    1.0);
    ProvidedMarkerPositions const providedPositions{-1, -1, 0,
                                                     1, -1, 0,
                                                     2,  0, 0,
                                                     1,  1, 0,
                                                    -1,  1, 0,
                                                    -2,  -0.5, 0};
    // clang-format on
    double constexpr knownMarkerDiameter{0.032};
    auto const focalLength{cameraMatrix.at<double>(0, 0)};
    PixelPosition const imageCenter{cameraMatrix.at<double>(0, 2),
                                    cameraMatrix.at<double>(1, 2)};
    auto constexpr PIX_DIST{100.0};
    auto const CENTER{cameraMatrix.at<double>(0, 2)};
    auto const Z0{focalLength / PIX_DIST};

    auto const MINOR{sphereToEllipseWidthHeight({0, 0, Z0}, focalLength,
                                                knownMarkerDiameter / 2)
                         .width};

    // clang-format off
    std::vector<hpm::Ellipse> marks{{{{CENTER - PIX_DIST, CENTER - PIX_DIST},
    0.0, MINOR, 0.0},
                 {{CENTER - 2*PIX_DIST, CENTER + 0.5*PIX_DIST}, 0.0, MINOR,
                 0.0},
                 {{CENTER + PIX_DIST, CENTER - PIX_DIST}, 0.0, MINOR, 0.0},
                 {{CENTER - PIX_DIST, CENTER + PIX_DIST}, 0.0, MINOR, 0.0},
                 {{CENTER + PIX_DIST, CENTER + PIX_DIST}, 0.0, MINOR, 0.0},
                 {{CENTER + 2*PIX_DIST, CENTER}, 0.0, MINOR, 0.0}}};
    auto const marksCpy{marks};
    // clang-format on

    double const err = identify(marks, knownMarkerDiameter, providedPositions,
                                focalLength, imageCenter, MarkerType::SPHERE);
    expect(marks[0] == marksCpy[3]);
    expect(marks[1] == marksCpy[4]);
    expect(marks[2] == marksCpy[5]);
    expect(marks[3] == marksCpy[2]);
    expect(marks[4] == marksCpy[0]);
    expect(marks[5] == marksCpy[1]);
    expect(err < EPS11);
  };

  double const focalLength{3000.0};
  double const markerRadius{70.0 / 2};
  PixelPosition const imageCenter{10000, 10000};

  skip / "compute ellipse equation of centered flat circle"_test = [&] {
    double constexpr r{105.0};
    double constexpr diameter{2.0 * r};
    double constexpr rsq{r * r};
    Ellipse const ellipse{imageCenter, diameter, diameter, 0.0};

    auto const [A, B, C, D, E, F] = ellipseEqInCamCoords(ellipse, imageCenter);

    expect(std::abs(A - 1.0 / rsq) < EPS9);
    expect(std::abs(B) < EPS9);
    expect(std::abs(C - 1.0 / rsq) < EPS9);
    expect(std::abs(D) < EPS9);
    expect(std::abs(E) < EPS9);
    expect(std::abs(F + 1) < EPS9);
  };

  skip /
      "compute expected ellipse equation of centered flat circle"_test = [&] {
    double constexpr r{105.0};
    double constexpr diameter{2.0 * r};
    double constexpr rsq{r * r};
    Ellipse const ellipse{imageCenter, diameter, diameter, 0.0};

    auto const [A, B, C, D, E, F] = ellipseEqInCamCoords2(ellipse, imageCenter);

    expect(std::abs(A - 1.0 / rsq) < EPS9);
    expect(std::abs(B) < EPS9);
    expect(std::abs(C - 1.0 / rsq) < EPS9);
    expect(std::abs(D) < EPS9);
    expect(std::abs(E) < EPS9);
    expect(std::abs(F + 1) < EPS9);
  };

  skip / "compute ellipse equation of x-offset flat circle"_test = [&] {
    double constexpr r{105.0};
    double constexpr diameter{2.0 * r};
    Ellipse const ellipse{imageCenter + PixelPosition{r, 0.0}, diameter,
                          diameter, 0.0};

    auto const [A, B, C, D, E, F] = ellipseEqInCamCoords(ellipse, imageCenter);
    std::array<double, 6> normalizedCopy{A / A, B / A, C / A,
                                         D / A, E / A, F / A};
    double const An{normalizedCopy[0]};
    double const Bn{normalizedCopy[1]};
    double const Cn{normalizedCopy[2]};
    double const Dn{normalizedCopy[3]};
    double const En{normalizedCopy[4]};
    double const Fn{normalizedCopy[5]};

    expect(std::abs(An - 1.0) < EPS9);
    expect(std::abs(Bn - 0.0) < EPS9);
    expect(std::abs(Cn - 1.0) < EPS9);
    expect(std::abs(Dn - (-2.0 * r)) < EPS9);
    expect(std::abs(En - 0.0) < EPS9);
    expect(std::abs(Fn - 0.0) < EPS9);
  };

  skip /
      "compute expected ellipse equation of x-offset flat circle"_test = [&] {
    double constexpr r{105.0};
    double constexpr diameter{2.0 * r};
    Ellipse const ellipse{imageCenter + PixelPosition{r, 0.0}, diameter,
                          diameter, 0.0};

    auto const [A, B, C, D, E, F] = ellipseEqInCamCoords2(ellipse, imageCenter);
    std::array<double, 6> normalizedCopy{A / A, B / A, C / A,
                                         D / A, E / A, F / A};
    double const An{normalizedCopy[0]};
    double const Bn{normalizedCopy[1]};
    double const Cn{normalizedCopy[2]};
    double const Dn{normalizedCopy[3]};
    double const En{normalizedCopy[4]};
    double const Fn{normalizedCopy[5]};

    expect(std::abs(An - 1.0) < EPS9);
    expect(std::abs(Bn - 0.0) < EPS9);
    expect(std::abs(Cn - 1.0) < EPS9);
    expect(std::abs(Dn - (-2 * r)) < EPS9);
    expect(std::abs(En - 0.0) < EPS9);
    expect(std::abs(Fn - 0.0) < EPS9);
  };

  skip / "compute ellipse equation of y-offset flat circle"_test = [&] {
    double constexpr r{105.0};
    double constexpr diameter{2.0 * r};
    Ellipse const ellipse{imageCenter + PixelPosition{0.0, 1.0}, diameter,
                          diameter, 0.0};

    auto const [A, B, C, D, E, F] = ellipseEqInCamCoords(ellipse, imageCenter);
    std::array<double, 6> normalizedCopy{A / A, B / A, C / A,
                                         D / A, E / A, F / A};
    double const An{normalizedCopy[0]};
    double const Bn{normalizedCopy[1]};
    double const Cn{normalizedCopy[2]};
    double const Dn{normalizedCopy[3]};
    double const En{normalizedCopy[4]};
    double const Fn{normalizedCopy[5]};

    expect(std::abs(An - 1.0) < EPS9);
    expect(std::abs(Bn - 0.0) < EPS9);
    expect(std::abs(Cn - 1.0) < EPS9);
    expect(std::abs(Dn - 0.0) < EPS9);
    expect(std::abs(En - (-2.0)) < EPS9);
    expect(std::abs(Fn - (1 - r * r)) < EPS9);
  };

  skip / "compute ellipse equation of simple x-ellipse"_test = [&] {
    double constexpr a{105.0};
    double constexpr b{50.0};

    Ellipse const ellipse{imageCenter, 2 * a, 2 * b, 0.0};

    auto const [A, B, C, D, E, F] = ellipseEqInCamCoords(ellipse, imageCenter);

    expect(std::abs(A - (1.0 / (a * a))) < EPS9);
    expect(std::abs(B - 0.0) < EPS9);
    expect(std::abs(C - (1.0 / (b * b))) < EPS9);
    expect(std::abs(D - 0.0) < EPS9);
    expect(std::abs(E - 0.0) < EPS9);
    expect(std::abs(F - (-1)) < EPS9);
  };

  skip / "compute ellipse equation of simple y-ellipse"_test = [&] {
    double constexpr a{105.0};
    double constexpr b{50.0};

    Ellipse const ellipse{imageCenter, 2 * a, 2 * b, CV_PI / 2.0};

    auto const [A, B, C, D, E, F] = ellipseEqInCamCoords(ellipse, imageCenter);

    expect(std::abs(A - (1.0 / (b * b))) < EPS9);
    expect(std::abs(B - 0.0) < EPS9);
    expect(std::abs(C - (1.0 / (a * a))) < EPS9);
    expect(std::abs(D - 0.0) < EPS9);
    expect(std::abs(E - 0.0) < EPS9);
    expect(std::abs(F - (-1)) < EPS9);
  };

  skip / "compute ellipse equation of 30-deg rotated ellipse"_test = [&] {
    double constexpr a{105.0};
    double constexpr b{50.0};

    Ellipse const ellipse{imageCenter, 2 * a, 2 * b, CV_PI / 6.0};

    double constexpr expectedA{3.0 / (4.0 * a * a) + 1.0 / (4.0 * b * b)};
    double constexpr expectedB{sqrt(3.0) / (2.0 * a * a) -
                               sqrt(3.0) / (2.0 * b * b)};
    double constexpr expectedC{1.0 / (4.0 * a * a) + 3.0 / (4.0 * b * b)};

    auto const [A, B, C, D, E, F] = ellipseEqInCamCoords(ellipse, imageCenter);

    expect(std::abs(A - (expectedA)) < EPS9);
    expect(std::abs(B - (expectedB)) < EPS3);
    expect(std::abs(C - (expectedC)) < EPS9);
    expect(std::abs(D - 0.0) < EPS9);
    expect(std::abs(E - 0.0) < EPS9);
    expect(std::abs(F - (-1)) < EPS9);
  };

  skip / "compute ellipse equation of xy-offsdet 30-deg rotated ellipse"_test =
      [&] {
        double constexpr a{105.0};
        double constexpr b{50.0};

        Ellipse const ellipse{imageCenter + PixelPosition{1.0, 2.0}, 2 * a,
                              2 * b, CV_PI / 6.0};

        double constexpr expectedA{3.0 / (4.0 * a * a) + 1.0 / (4.0 * b * b)};
        double constexpr expectedB{sqrt(3.0) / (2.0 * a * a) -
                                   sqrt(3.0) / (2.0 * b * b)};
        double constexpr expectedC{1.0 / (4.0 * a * a) + 3.0 / (4.0 * b * b)};
        double constexpr expectedD{-3.0 / (2.0 * a * a) - sqrt(3.0) / (a * a) -
                                   1.0 / (2.0 * b * b) + sqrt(3.0) / (b * b)};
        double constexpr expectedE{-sqrt(3.0) / (2 * a * a) - 1.0 / (a * a) +
                                   sqrt(3.0) / (2 * b * b) - 3.0 / (b * b)};
        double constexpr expectedF{3.0 / (4.0 * a * a) + sqrt(3.0) / (a * a) +
                                   1.0 / (a * a) + 1.0 / (4 * b * b) -
                                   sqrt(3) / (b * b) + 3.0 / (b * b) - 1.0};

        auto const [A, B, C, D, E, F] =
            ellipseEqInCamCoords(ellipse, imageCenter);

        expect(std::abs(A - (expectedA)) < EPS9);
        expect(std::abs(B - (expectedB)) < EPS3);
        expect(std::abs(C - (expectedC)) < EPS9);
        expect(std::abs(D - (expectedD)) < EPS9);
        expect(std::abs(E - (expectedE)) < EPS9);
        expect(std::abs(F - (expectedF)) < EPS9);
      };

  skip / "compute ellipse equation of more xy-offset ellipse"_test = [&] {
    double constexpr a{105.0};
    double constexpr b{50.0};

    Ellipse const ellipse{
        imageCenter + PixelPosition{imageCenter.x / 2.0, imageCenter.y / 2.0},
        2 * a, 2 * b, 0.0};

    auto [expA, expB, expC, expD, expE, expF] =
        ellipseEqInCamCoords2(ellipse, imageCenter);
    auto const [A, B, C, D, E, F] = ellipseEqInCamCoords(ellipse, imageCenter);

    expect(std::abs(A - expA) < EPS9);
    expect(std::abs(B - expB) < EPS3);
    expect(std::abs(C - expC) < EPS9);
    expect(std::abs(D - expD) < EPS9);
    expect(std::abs(E - expE) < EPS9);
    expect(std::abs(F - expF) < EPS9);
  };

  skip / "compute ellipse equation of more xy-offset and rotated ellipse"_test =
      [&] {
        double constexpr a{105.0};
        double constexpr b{50.0};

        Ellipse const ellipse{imageCenter + PixelPosition{imageCenter.x / 2.0,
                                                          imageCenter.y / 2.0},
                              2 * a, 2 * b, -CV_PI * 1.0 / 3.0};

        auto [expA, expB, expC, expD, expE, expF] =
            ellipseEqInCamCoords2(ellipse, imageCenter);
        auto const [A, B, C, D, E, F] =
            ellipseEqInCamCoords(ellipse, imageCenter);

        std::cout << A << ' ' << expA << '\n';
        std::cout << B << ' ' << expB << '\n';
        std::cout << C << ' ' << expC << '\n';
        std::cout << D << ' ' << expD << '\n';
        std::cout << E << ' ' << expE << '\n';
        std::cout << F << ' ' << expF << '\n';
        expect(std::abs(A - expA) < EPS9);
        expect(std::abs(B - expB) < EPS3);
        expect(std::abs(C - expC) < EPS9);
        expect(std::abs(D - expD) < EPS9);
        expect(std::abs(E - expE) < EPS9);
        expect(std::abs(F - expF) < EPS9);
      };

  skip / "center flat disk position"_test = [&] {
    double const zDist{1000.0};
    double const projectionDiameter{2 * markerRadius * focalLength / zDist};
    auto const gotPosition = toPosition(
        Ellipse{imageCenter, projectionDiameter, projectionDiameter, 0.0},
        markerRadius * 2, focalLength, imageCenter, MarkerType::DISK);

    expect(std::abs(gotPosition.x - 0.0) < EPS9);
    expect(std::abs(gotPosition.y - 0.0) < EPS9);
    expect(std::abs(gotPosition.z - 1000.0) < EPS9);
  };

  skip / "x-offset flat disk position"_test = [&] {
    double const zDist{1000.0};
    double const projectionDiameter{2 * markerRadius * focalLength / zDist};
    auto const gotPosition = toPosition(
        Ellipse{imageCenter + PixelPosition{projectionDiameter / 2, 0},
                projectionDiameter, projectionDiameter, 0.0},
        markerRadius * 2, focalLength, imageCenter, MarkerType::DISK);

    expect(std::abs(gotPosition.x - markerRadius) < EPS9);
    expect(std::abs(gotPosition.y - 0.0) < EPS9);
    expect(std::abs(gotPosition.z - 1000.0) < EPS9);
  };

  "xy-offset flat disk position"_test = [&] {
    double const zDist{1000.0};
    double const projectionDiameter{2 * markerRadius * focalLength / zDist};
    auto const gotPosition = toPosition(
        Ellipse{imageCenter + PixelPosition{projectionDiameter * 3,
                                            projectionDiameter * 3},
                projectionDiameter, projectionDiameter, 0.0},
        markerRadius * 2, focalLength, imageCenter, MarkerType::DISK);

    std::cout << gotPosition << '\n';
    expect(std::abs(gotPosition.x - (markerRadius * 6)) < EPS9);
    expect(std::abs(gotPosition.y - (markerRadius * 6)) < EPS9);
    expect(std::abs(gotPosition.z - 1000.0) < EPS9);
  };

  // clang-format off
  cv::Mat const openScadCameraMatrix = (cv::Mat_<double>(3, 3) << 2 * 3375.85,        0.00, 2 * 1280.0,
                                                                         0.00, 2 * 3375.85, 2 *  671.5,
                                                                         0.00,        0.00,        1.0);
  // clang-format on
  skip / "center flat disk position from render"_test = [&] {
    std::string const imageFileName{hpm::getPath(
        "test-images/central_flat_disk_d70_0_0_0_0_0_0_1000_5120_2686.png")};
    cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
    expect((not image.empty()) >> fatal);
    auto const &cam = openScadCameraMatrix;
    double const focalLength{
        std::midpoint(cam.at<double>(0, 0), cam.at<double>(1, 1))};
    PixelPosition const imageCenter{cam.at<double>(0, 2), cam.at<double>(1, 2)};
    auto const marks{rawEllipseDetect(image, false)};
    expect((marks.size() == 1) >> fatal);
    auto const gotPosition = toPosition(marks[0], markerRadius * 2, focalLength,
                                        imageCenter, MarkerType::DISK);

    expect(std::abs(gotPosition.x - 0.0) < EPS4);
    expect(std::abs(gotPosition.y - 0.0) < EPS4);
    expect(std::abs(gotPosition.z - 1000.0) < 1.0_d);
  };

  skip / "center sphere position"_test = [&] {
    double const zDist{1000.0};

    double const gamma1{asin(markerRadius / zDist)};
    double const closerZ{zDist - markerRadius * sin(gamma1)};
    double const closerR{markerRadius * cos(gamma1)};
    double const projectionHeight{focalLength * 2 * closerR / closerZ};

    auto const gotPosition = toPosition(
        Ellipse{imageCenter, projectionHeight, projectionHeight, 0.0},
        markerRadius * 2, focalLength, imageCenter, MarkerType::SPHERE);

    expect(std::abs(gotPosition.x - 0.0) < EPS9);
    expect(std::abs(gotPosition.y - 0.0) < EPS9);
    expect(std::abs(gotPosition.z - 1000.0) < EPS9);
  };

  skip / "x-offset sphere position"_test = [&] {
    CameraFramedPosition const knownPos{10.0, 0.0, 1000.0};
    auto const [width, height, xt, yt] =
        sphereToEllipseWidthHeight(knownPos, focalLength, markerRadius);

    auto const gotPosition = toPosition(
        Ellipse{imageCenter + PixelPosition{xt, yt}, width, height, 0.0},
        markerRadius * 2, focalLength, imageCenter, MarkerType::SPHERE);

    expect(std::abs(gotPosition.x - 10.0) < EPS9);
    expect(std::abs(gotPosition.y - 0.0) < EPS9);
    expect(std::abs(gotPosition.z - 1000.0) < EPS9);
  };

  skip / "y-offset sphere position"_test = [&] {
    CameraFramedPosition const knownPos{0.0, 10.0, 1000.0};
    auto const [width, height, xt, yt] =
        sphereToEllipseWidthHeight(knownPos, focalLength, markerRadius);

    auto const gotPosition = toPosition(
        Ellipse{imageCenter + PixelPosition{xt, yt}, width, height, M_PI / 2},
        markerRadius * 2, focalLength, imageCenter, MarkerType::SPHERE);

    expect(std::abs(gotPosition.x - 0.0) < EPS9);
    expect(std::abs(gotPosition.y - 10.0) < EPS9);
    expect(std::abs(gotPosition.z - 1000.0) < EPS9);
  };

  skip / "xy-offset sphere positions"_test = [&] {
    for (double dxy{5.0}; dxy <= 1000.0; dxy = dxy + 10.0) {
      for (double ang{0.0}; ang < 2 * M_PI; ang += M_PI / 6.0) {
        double const xDist{dxy * cos(ang)};
        double const yDist{dxy * sin(ang)};
        double const zDist{1000.0};
        CameraFramedPosition const knownPos{xDist, yDist, zDist};
        auto const [width, height, xt, yt] =
            sphereToEllipseWidthHeight(knownPos, focalLength, markerRadius);

        auto const gotPosition = toPosition(
            Ellipse{imageCenter + PixelPosition{xt, yt}, width, height, ang},
            markerRadius * 2, focalLength, imageCenter, MarkerType::SPHERE);

        auto constexpr EPS{0.00000000115_d}; // 1.15e-9 precision
        expect(std::abs(gotPosition.x - xDist) < EPS);
        expect(std::abs(gotPosition.y - yDist) < EPS);
        expect(std::abs(gotPosition.z - zDist) < EPS);
      }
    }
  };

  skip / "x-offset flat disk position"_test = [&] {
    double const zDist{1000.0};
    double const projectionWidth{2 * markerRadius * focalLength / zDist};
    double const projectionHeight{projectionWidth};
    auto const gotPosition = toPosition(
        Ellipse{imageCenter +
                    PixelPosition{markerRadius * focalLength / zDist, 0},
                projectionWidth, projectionHeight, 0.0},
        markerRadius * 2, focalLength, imageCenter, MarkerType::DISK);

    expect(std::abs(gotPosition.x - markerRadius) < EPS9);
    expect(std::abs(gotPosition.y - 0.0) < EPS9);
    expect(std::abs(gotPosition.z - 1000.0) < EPS9);
  };

  skip / "x-offset flat disk position from render"_test = [&] {
    std::string const imageFileName{hpm::getPath(
        "test-images/x_offset_flat_disk_d70_0_0_0_0_0_0_1000_5120_2686.png")};
    cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
    expect((not image.empty()) >> fatal);
    auto const &cam = openScadCameraMatrix;
    double const focalLength{
        std::midpoint(cam.at<double>(0, 0), cam.at<double>(1, 1))};
    PixelPosition const imageCenter{cam.at<double>(0, 2), cam.at<double>(1, 2)};
    auto const marks{rawEllipseDetect(image, false)};
    expect((marks.size() == 1) >> fatal);
    auto const gotPosition = toPosition(marks[0], markerRadius * 2, focalLength,
                                        imageCenter, MarkerType::DISK);

    expect(std::abs(gotPosition.x - markerRadius) < EPS2);
    expect(std::abs(gotPosition.y - 0.0) < EPS2);
    expect(std::abs(gotPosition.z - 1000.0) < 1);
  };

  skip / "center 45-deg disk position from render"_test = [&] {
    std::string const imageFileName{hpm::getPath(
        "test-images/central_45_deg_disk_d70_0_0_0_0_0_0_1000_5120_2686.png")};
    cv::Mat const image = cv::imread(imageFileName, cv::IMREAD_COLOR);
    expect((not image.empty()) >> fatal);
    auto const &cam = openScadCameraMatrix;
    double const focalLength{
        std::midpoint(cam.at<double>(0, 0), cam.at<double>(1, 1))};
    PixelPosition const imageCenter{cam.at<double>(0, 2), cam.at<double>(1, 2)};
    auto const marks{rawEllipseDetect(image, false)};
    expect((marks.size() == 1) >> fatal);
    auto const gotPosition = toPosition(marks[0], markerRadius * 2, focalLength,
                                        imageCenter, MarkerType::DISK);
    std::cout << "gotPosition=" << gotPosition << std::endl;

    expect(std::abs(gotPosition.x - 0.0) < EPS4);
    expect(std::abs(gotPosition.y - 0.0) < EPS4);
    expect(std::abs(gotPosition.z - 1000.0) < 1);
  };

  return 0;
}
