#pragma once

#include <hpm/ellipse.h++>
#include <hpm/simple-types.h++>

#include <vector>

namespace hpm {

std::array<double, 6> ellipseEqInCamCoords(hpm::Ellipse const &ellipse,
                                           PixelPosition const &imageCenter);

CameraFramedPosition toPosition(Ellipse const &markerProjection,
                                double markerDiameter, double focalLength,
                                PixelPosition const &imageCenter,
                                MarkerType markerType);

CameraFramedPosition sphereProjToPosition(Ellipse const &sphereProjection,
                                          double sphereDiameter,
                                          double focalLength,
                                          PixelPosition const &imageCenter);

CameraFramedPosition diskProjToPosition(Ellipse const &diskProjection,
                                        double diskDiameter, double focalLength,
                                        PixelPosition const &imageCenter);

PixelPosition centerRay(Ellipse const &markerProjection, double markerDiameter,
                        double focalLength, PixelPosition const &imageCenter,
                        MarkerType markerType);

PixelPosition sphereCenterRay(Ellipse const &sphereProjection,
                              double sphereDiameter, double focalLength,
                              PixelPosition const &imageCenter);

PixelPosition diskCenterRay(Ellipse const &diskProjection, double diskDiameter,
                            double focalLength,
                            PixelPosition const &imageCenter);

double identify(std::vector<Ellipse> &marks, double markerDiameter,
                ProvidedMarkerPositions const &markPos, double focalLength,
                PixelPosition const &imageCenter, MarkerType markerType);
} // namespace hpm
