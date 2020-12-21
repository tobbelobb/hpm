#include <limits>

#include <boost/ut.hpp> //import boost.ut;

#include <hpm/solve-pnp.h++>

using namespace hpm;

auto main() -> int {
  using namespace boost::ut;
  // clang-format off
  cv::Mat const cameraMatrix = (cv::Mat_<double>(3, 3) << 3000.0,    0.0, 1000.0,
                                                             0.0, 3000.0, 1000.0,
                                                             0.0,    0.0,    1.0);
  ProvidedMarkerPositions const providedPositions{-1, -1, 0,
                                                   1, -1, 0,
                                                   1,  0, 0,
                                                   1,  1, 0,
                                                  -1,  1, 0,
                                                  -1,  0, 0};
  // clang-format on

  auto constexpr PIX_DIST{100.0};
  auto const CENTER{cameraMatrix.at<double>(0, 2)};
  auto const F{cameraMatrix.at<double>(0, 0)};
  auto const X0{F / PIX_DIST};

  "No rotation"_test = [&]() {
    IdentifiedHpMarks const identifiedMarks{
        {CENTER - PIX_DIST, CENTER - PIX_DIST},
        {CENTER + PIX_DIST, CENTER - PIX_DIST},
        {CENTER + PIX_DIST, CENTER},
        {CENTER + PIX_DIST, CENTER + PIX_DIST},
        {CENTER - PIX_DIST, CENTER + PIX_DIST},
        {CENTER - PIX_DIST, CENTER}};

    std::optional<SixDof> const result{
        solvePnp(cameraMatrix, providedPositions, identifiedMarks)};
    expect((result.has_value()) >> fatal); // NOLINT
    auto const &value{result.value()};

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

    auto const ang{0.1};
    auto const sin{std::sin(ang)};
    auto const cos{std::cos(ang)};
    PixelPosition const center{CENTER, CENTER};

    for (auto &mark : marks) {
      mark.x = (mark.x - CENTER) * cos - (mark.y - CENTER) * sin + CENTER;
      mark.y = (mark.x - CENTER) * sin + (mark.y - CENTER) * cos + CENTER;
    }

    IdentifiedHpMarks const identifiedMarks{marks};

    std::optional<SixDof> const result{
        solvePnp(cameraMatrix, providedPositions, identifiedMarks)};
    expect((result.has_value()) >> fatal); // NOLINT
    auto const &value{result.value()};

    /* Right above is a particularly sensitive angle that makes the PnP problem
     * a bit unstable, so allow quite large errors. */
    auto constexpr EPS{0.13_d};
    expect(abs(value.rotX()) < EPS) << "X rotation";
    expect(abs(value.rotY()) < EPS) << "Y rotation";
    expect(abs(value.rotZ() - 0.1) < EPS) << "Z rotation";
    expect(abs(value.x()) < EPS) << "X translation";
    expect(abs(value.y()) < EPS) << "Y translation";
    expect(abs(value.z() - static_cast<double>(X0)) < EPS) << "Z translation";
    expect(value.reprojectionError < 0.45_d) << "Reprojection error";
  };

  "Pi over four Y-rotation from right above"_test = [&]() {
    auto const sqrt2Inv{1.0 / std::sqrt(2.0)};
    // Some markers get closer to the camera by such a rotation
    // Instead of PIX_DIST, they will have the following X
    // and Y offsets from the images center
    auto const closersX{F * sqrt2Inv / (X0 - sqrt2Inv)};
    auto const closersY{F / (X0 - sqrt2Inv)};
    // Some markers are moved farther away from the camera
    // by the rotation
    auto const farthersX{F * sqrt2Inv / (X0 + sqrt2Inv)};
    auto const farthersY{F / (X0 + sqrt2Inv)};

    IdentifiedHpMarks const identifiedMarks{
        {CENTER - closersX, CENTER - closersY},
        {CENTER + farthersX, CENTER - farthersY},
        {CENTER + farthersX, CENTER},
        {CENTER + farthersX, CENTER + farthersY},
        {CENTER - closersX, CENTER + closersY},
        {CENTER - closersX, CENTER}};

    std::optional<SixDof> const result{
        solvePnp(cameraMatrix, providedPositions, identifiedMarks)};
    expect((result.has_value()) >> fatal); // NOLINT
    auto const &value{result.value()};

    auto constexpr EPS{0.0000051_d};
    expect(abs(value.rotX()) < EPS) << "X rotation";
    expect(abs(value.rotY() + CV_PI / 4.0) < EPS) << "Y rotation";
    expect(abs(value.rotZ()) < EPS) << "Z rotation";
    expect(abs(value.x()) < EPS) << "X translation";
    expect(abs(value.y()) < EPS) << "Y translation";
    expect(abs(value.z() - static_cast<double>(X0)) < EPS) << "Z translation";
    expect(value.reprojectionError < EPS) << "Reprojection error";
  };

  "Only five found markers"_test = [&]() {
    auto const sqrt2Inv{1.0 / std::sqrt(2.0)};
    auto const closersX{F * sqrt2Inv / (X0 - sqrt2Inv)};
    auto const closersY{F / (X0 - sqrt2Inv)};
    auto const farthersX{F * sqrt2Inv / (X0 + sqrt2Inv)};
    auto const farthersY{F / (X0 + sqrt2Inv)};
    IdentifiedHpMarks identifiedMarks{{CENTER - closersX, CENTER - closersY},
                                      {CENTER + farthersX, CENTER - farthersY},
                                      {CENTER + farthersX, CENTER},
                                      {CENTER + farthersX, CENTER + farthersY},
                                      {CENTER - closersX, CENTER + closersY},
                                      {CENTER - closersX, CENTER}};
    identifiedMarks.m_pixelPositions[0] = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()};
    identifiedMarks.m_identified[0] = false;

    std::optional<SixDof> const result{
        solvePnp(cameraMatrix, providedPositions, identifiedMarks)};
    expect((result.has_value()) >> fatal); // NOLINT

    auto const &value{result.value()};
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
