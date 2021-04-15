#pragma once

#include <hpm/simple-types.h++>

namespace hpm {

struct SixDof {
  Vector3d rotation{0, 0, 0};
  Vector3d translation{0, 0, 0};
  double reprojectionError{0};

  double x() const { return translation(0); }
  double y() const { return translation(1); }
  double z() const { return translation(2); }
  double rotX() const { return rotation(0); }
  double rotY() const { return rotation(1); }
  double rotZ() const { return rotation(2); }

  friend std::ostream &operator<<(std::ostream &out, SixDof const &sixDof) {
    return out << sixDof.rotation << '\n'
               << sixDof.translation << '\n'
               << sixDof.reprojectionError;
  };

  bool operator<(SixDof const &other) {
    return reprojectionError < other.reprojectionError;
  }
};

} // namespace hpm
