#include <hpm/ellipse.h++>

using namespace hpm;

hpm::Ellipse::Ellipse(ed::mEllipse const &edEllipse)
    : m_center(edEllipse.center) {
  if (edEllipse.axes.width >= edEllipse.axes.height) {
    m_major = 2.0 * edEllipse.axes.width;
    m_minor = 2.0 * edEllipse.axes.height;
    m_rot = edEllipse.theta;
  } else {
    m_major = 2.0 * edEllipse.axes.height;
    m_minor = 2.0 * edEllipse.axes.width;
    if (edEllipse.theta > 0.0) {
      m_rot = edEllipse.theta - M_PI / 2.0;
    } else {
      m_rot = edEllipse.theta + M_PI / 2.0;
    }
  }
}
