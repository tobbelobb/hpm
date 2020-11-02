#pragma once

#include <ostream>
#include <vector>

#include <opencv2/core.hpp>

struct Position {
  double x{0};
  double y{0};
  double z{0};

  friend std::ostream &operator<<(std::ostream &os, Position const &position) {
    return os << "(" << position.x << ", " << position.y << ", " << position.z
              << ')';
  }
};

struct Marker {
  double x{0};
  double y{0};
  double r{0};

  friend std::ostream &operator<<(std::ostream &os, Marker const &marker) {
    return os << "(" << marker.x << ", " << marker.y << ", " << marker.r << ')';
  }
};

struct CamParams {
  cv::Mat intrinsic{0};
  cv::Mat distortion{0};

  friend std::ostream &operator<<(std::ostream &os,
                                  CamParams const &camParams) {
    return os << "Intrinsic: " << camParams.intrinsic
              << " Distortion: " << camParams.distortion;
  }
};

std::vector<Marker> detectMarkers(CamParams const &camParams,
                                  cv::InputArray const undistortedImage,
                                  bool showIntermediateImages);

void draw(/* inout */ cv::Mat image, Marker const &marker);
