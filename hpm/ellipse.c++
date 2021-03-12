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
  m_equation = edEllipse.equation;
}

// Test this...
ed::EllipseEquation hpm::Ellipse::computeEq() const {
  cv::Matx22d const R_e(cos(m_rot), -sin(m_rot), sin(m_rot), cos(m_rot));
  cv::Matx22d const temp(1.0 / ((m_major / 2) * (m_major / 2)), 0.0, 0.0,
                         1.0 / ((m_minor / 2) * (m_minor / 2)));
  auto const M = R_e * (temp * R_e.t());
  cv::Matx21d const X_0(m_center);

  return {M(0, 0),
          M(0, 1),
          M(1, 1),
          M(0, 0) * m_center.x + M(0, 1) * m_center.y,
          M(0, 1) * m_center.x + M(1, 1) * m_center.y,
          (X_0.t() * (M * X_0))(0, 0) - 1.0};
}
