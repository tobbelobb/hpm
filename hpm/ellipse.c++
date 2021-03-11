#include <hpm/ellipse.h++>

using namespace hpm;

hpm::Ellipse::Ellipse(mEllipse const &ellipse) : m_center(ellipse.center) {
  if (ellipse.axes.width >= ellipse.axes.height) {
    m_major = 2.0 * ellipse.axes.width;
    m_minor = 2.0 * ellipse.axes.height;
    m_rot = ellipse.theta;
  } else {
    m_major = 2.0 * ellipse.axes.height;
    m_minor = 2.0 * ellipse.axes.width;
    if (ellipse.theta > 0.0) {
      m_rot = ellipse.theta - M_PI / 2.0;
    } else {
      m_rot = ellipse.theta + M_PI / 2.0;
    }
  }
}
