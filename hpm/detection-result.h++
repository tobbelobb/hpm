#pragma once

#include <vector>

#include <hpm/simple-types.h++>

namespace hpm {

struct DetectionResult {
  std::vector<hpm::KeyPoint> red;
  std::vector<hpm::KeyPoint> green;
  std::vector<hpm::KeyPoint> blue;

  size_t size() const { return red.size() + green.size() + blue.size(); }
};

} // namespace hpm
