#include <boost/ut.hpp> //import boost.ut;

#include <hpm/solve-pnp.h++>

auto main() -> int {
  using namespace boost::ut;
  // clang-format off
  cv::Mat const cameraMatrix = (cv::Mat_<double>(3, 3) << 3000.0,    0.0, 1000.0,
                                                             0.0, 3000.0, 1000.0,
                                                             0.0,    0.0,    1.0);
  std::vector<cv::Point3d> const markerPositions{
      {-1, -1, 0}, {1, -1, 0}, {1, 0, 0}, {1, 1, 0}, {-1, 1, 0}, {-1, 0, 0}};
  // clang-format on
  float constexpr PIX_DIST{100};
  float const CENTER{static_cast<float>(cameraMatrix.at<double>(0, 2))};
  float const F{static_cast<float>(cameraMatrix.at<double>(0, 0))};
  float const X0{F / PIX_DIST};

  "No rotation"_test = [&]() {
    IdentifiedHpMarks const identifiedMarks{
        .red0 = {CENTER - PIX_DIST, CENTER - PIX_DIST},
        .red1 = {CENTER + PIX_DIST, CENTER - PIX_DIST},
        .green0 = {CENTER + PIX_DIST, CENTER},
        .green1 = {CENTER + PIX_DIST, CENTER + PIX_DIST},
        .blue0 = {CENTER - PIX_DIST, CENTER + PIX_DIST},
        .blue1 = {CENTER - PIX_DIST, CENTER}};

    std::optional<SixDof> const result{
        solvePnp(cameraMatrix, markerPositions, identifiedMarks)};
    expect((result.has_value()) >> fatal); // NOLINT
    auto const value{result.value()};

    auto constexpr EPS{0.000001_d};
    expect(abs(value.rotX()) < EPS) << "X rotation";
    expect(abs(value.rotY()) < EPS) << "Y rotation";
    expect(abs(value.rotZ()) < EPS) << "Z rotation";
    expect(abs(value.x()) < EPS) << "X translation";
    expect(abs(value.y()) < EPS) << "Y translation";
    expect(abs(value.z() - static_cast<double>(X0)) < EPS) << "Z translation";
    expect(value.reprojectionError < EPS) << "Reprojection error";
  };

  "Small Z-rotation from right above"_test = [&]() {
    std::array<PixelPosition, 6> marks{{{CENTER - PIX_DIST, CENTER - PIX_DIST},
                                        {CENTER + PIX_DIST, CENTER - PIX_DIST},
                                        {CENTER + PIX_DIST, CENTER},
                                        {CENTER + PIX_DIST, CENTER + PIX_DIST},
                                        {CENTER - PIX_DIST, CENTER + PIX_DIST},
                                        {CENTER - PIX_DIST, CENTER}}};

    float const ang{0.1F};
    float const sin{std::sin(ang)};
    float const cos{std::cos(ang)};
    PixelPosition const center{CENTER, CENTER};

    for (auto &mark : marks) {
      mark.x = (mark.x - CENTER) * cos - (mark.y - CENTER) * sin + CENTER;
      mark.y = (mark.x - CENTER) * sin + (mark.y - CENTER) * cos + CENTER;
    }

    IdentifiedHpMarks const identifiedMarks{marks};

    std::optional<SixDof> const result{
        solvePnp(cameraMatrix, markerPositions, identifiedMarks)};
    expect((result.has_value()) >> fatal); // NOLINT
    auto const value{result.value()};

    /* Right above is a particularly sensitive angle that makes the PnP problem
     * a bit unstable, so allow quite large errors. */
    auto constexpr EPS{0.13_d};
    expect(abs(value.rotX()) < EPS) << "X rotation";
    expect(abs(value.rotY()) < EPS) << "Y rotation";
    expect(abs(value.rotZ() - 0.1) < EPS) << "Z rotation";
    expect(abs(value.x()) < EPS) << "X translation";
    expect(abs(value.y()) < EPS) << "Y translation";
    expect(abs(value.z() - static_cast<double>(X0)) < EPS) << "Z translation";
    expect(value.reprojectionError < 0.3_d) << "Reprojection error";
  };

  "Pi over four Y-rotation from right above"_test = [&]() {
    float const sqrt2Inv{1.0F / std::sqrt(2.0F)};
    // Some markers get closer to the camera by such a rotation
    // Instead of PIX_DIST, they will have the following X
    // and Y offsets from the images center
    float const closersX{F * sqrt2Inv / (X0 - sqrt2Inv)};
    float const closersY{F / (X0 - sqrt2Inv)};
    // Some markers are moved farther away from the camera
    // by the rotation
    float const farthersX{F * sqrt2Inv / (X0 + sqrt2Inv)};
    float const farthersY{F / (X0 + sqrt2Inv)};

    IdentifiedHpMarks const identifiedMarks{
        .red0 = {CENTER - closersX, CENTER - closersY},
        .red1 = {CENTER + farthersX, CENTER - farthersY},
        .green0 = {CENTER + farthersX, CENTER},
        .green1 = {CENTER + farthersX, CENTER + farthersY},
        .blue0 = {CENTER - closersX, CENTER + closersY},
        .blue1 = {CENTER - closersX, CENTER}};

    std::optional<SixDof> const result{
        solvePnp(cameraMatrix, markerPositions, identifiedMarks)};
    expect((result.has_value()) >> fatal); // NOLINT
    auto const value{result.value()};

    auto constexpr EPS{0.0000051_d};
    expect(abs(value.rotX()) < EPS) << "X rotation";
    expect(abs(value.rotY() + CV_PI / 4.0) < EPS) << "Y rotation";
    expect(abs(value.rotZ()) < EPS) << "Z rotation";
    expect(abs(value.x()) < EPS) << "X translation";
    expect(abs(value.y()) < EPS) << "Y translation";
    expect(abs(value.z() - static_cast<double>(X0)) < EPS) << "Z translation";
    expect(value.reprojectionError < EPS) << "Reprojection error";
  };
}
