#pragma once

#include <hpm/marks.h++>
#include <hpm/simple-types.h++>
#include <hpm/solve-pnp.h++>

#include <hpm/open-cv-warnings-disabler.h++>
DISABLE_WARNINGS
#include <opencv2/core.hpp>
ENABLE_WARNINGS

#include <iostream>
#include <vector>

namespace hpm {
struct MarkerParams {
  ProvidedMarkerPositions m_providedMarkerPositions{0.0};
  double m_diameter{0.0};
  PixelPosition m_topLeftMarkerCenter{0.0, 0.0};
  Mark::Type m_type{Mark::Type::SPHERE};
};

struct FinderConfig {
  bool m_showIntermediateImages{false};
  bool m_verbose{false};
  bool m_fitByDistance{false};
};
} // namespace hpm

hpm::Marks findMarks(cv::InputArray undistortedImage,
                     hpm::MarkerParams const &markerParams,
                     double const focalLength,
                     hpm::PixelPosition const &imageCenter,
                     hpm::FinderConfig const &config);

std::vector<hpm::CameraFramedPosition>
findIndividualMarkerPositions(hpm::Marks const &marks,
                              double knownMarkerDiameter, double focalLength,
                              hpm::PixelPosition const &imageCenter);
