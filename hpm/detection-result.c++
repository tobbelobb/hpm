#include <hpm/detection-result.h++>

using namespace hpm;

auto hpm::zFromSemiMinor(double markerR, double f, double semiMinor) -> double {
  double const rSmall = markerR * f / sqrt(semiMinor * semiMinor + f * f);
  double const thetaZ = atan(semiMinor / f);
  return rSmall * f / semiMinor + markerR * sin(thetaZ);
}

auto hpm::centerRayFromZ(double c, double markerR, double z) -> double {
  return c * (z * z - markerR * markerR) / (z * z);
}
