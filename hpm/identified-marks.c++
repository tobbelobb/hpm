
#include <hpm/identified-marks.h++>

using namespace hpm;

IdentifiedMarks::IdentifiedMarks(Marks const &marks, double const markerR,
                                 double const f,
                                 PixelPosition const &imageCenter) {
  if (marks.m_red.size() != 2 or marks.m_green.size() != 2 or
      marks.m_blue.size() != 2) {
    return;
  }

  std::vector<hpm::Mark> all{marks.getFlatCopy()};

  for (size_t i{0}; i < m_pixelPositions.size() and i < all.size(); ++i) {
    m_pixelPositions[i] = all[i].getCenterRay(markerR, f, imageCenter);
    m_identified[i] = true;
  }
}

bool IdentifiedMarks::isIdentified(size_t idx) const {
  return idx < m_identified.size() and m_identified[idx];
}

PixelPosition IdentifiedMarks::getPixelPosition(size_t idx) const {
  return m_pixelPositions[idx];
}

bool IdentifiedMarks::allIdentified() const {
  return std::all_of(m_identified.begin(), m_identified.end(), std::identity());
}

std::ostream &operator<<(std::ostream &out,
                         IdentifiedMarks const &identifiedMarks) {
  for (size_t i{0}; i < identifiedMarks.m_pixelPositions.size(); ++i) {
    if (identifiedMarks.isIdentified(i)) {
      out << identifiedMarks.m_pixelPositions[i];
    } else {
      out << '?';
    }
    out << '\n';
  }
  return out;
}
