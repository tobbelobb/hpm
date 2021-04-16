#pragma once

#include <hpm/ellipse.h++>
#include <hpm/simple-types.h++>

#include <vector>

namespace hpm {

std::array<double, 6> ellipseEqInCamCoords2(hpm::Ellipse const &ellipse,
                                            PixelPosition const &imageCenter);

std::array<double, 6> ellipseEqInCamCoords(hpm::Ellipse const &ellipse,
                                           PixelPosition const &imageCenter);

CameraFramedPosition toPosition(
    Ellipse const &markerProjection, double markerDiameter, double focalLength,
    PixelPosition const &imageCenter, MarkerType markerType,
    CameraFramedPosition const &expectedNormalDirection = {0.0, 0.0, 0.0});

CameraFramedPosition sphereProjToPosition(Ellipse const &sphereProjection,
                                          double sphereDiameter,
                                          double focalLength,
                                          PixelPosition const &imageCenter);

struct TwoPoses {
  CameraFramedPosition center0;
  CameraFramedPosition normal0;
  CameraFramedPosition center1;
  CameraFramedPosition normal1;
};

TwoPoses diskProjToTwoPoses(Ellipse const &diskProjection, double diskDiameter,
                            double focalLength,
                            PixelPosition const &imageCenter);

CameraFramedPosition
diskProjToPosition(Ellipse const &diskProjection, double diskDiameter,
                   double focalLength, PixelPosition const &imageCenter,
                   CameraFramedPosition const &expectedNormalDirection);

PixelPosition centerRay(Ellipse const &markerProjection, double markerDiameter,
                        double focalLength, PixelPosition const &imageCenter,
                        MarkerType markerType,
                        CameraFramedPosition const &expectedNormalDirection = {
                            0.0, 0.0, 0.0});

PixelPosition sphereCenterRay(Ellipse const &sphereProjection,
                              double sphereDiameter, double focalLength,
                              PixelPosition const &imageCenter);

PixelPosition
diskCenterRay(Ellipse const &diskProjection, double diskDiameter,
              double focalLength, PixelPosition const &imageCenter,
              CameraFramedPosition const &expectedNormalDirection);

double identify(std::vector<Ellipse> &marks, double markerDiameter,
                ProvidedMarkerPositions const &markPos, double focalLength,
                PixelPosition const &imageCenter, MarkerType markerType,
                bool tryHard = false);
} // namespace hpm
