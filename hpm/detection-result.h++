#pragma once

#include <vector>

#include <hpm/simple-types.h++>

namespace hpm {

struct DetectionResult {
  std::vector<hpm::KeyPoint> red;
  std::vector<hpm::KeyPoint> green;
  std::vector<hpm::KeyPoint> blue;

  size_t size() const { return red.size() + green.size() + blue.size(); }

  std::vector<hpm::KeyPoint> getFlatCopy() const {
    // pipes? join?
    std::vector<hpm::KeyPoint> all{};
    all.reserve(size());
    all.insert(all.end(), red.begin(), red.end());
    all.insert(all.end(), green.begin(), green.end());
    all.insert(all.end(), blue.begin(), blue.end());
    return all;
  }
};

} // namespace hpm
