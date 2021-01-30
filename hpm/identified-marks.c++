
#include <hpm/identified-marks.h++>

using namespace hpm;

static inline auto signed2DCross(PixelPosition const &v0,
                                 PixelPosition const &v1,
                                 PixelPosition const &v2) {
  return (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
}

static inline auto isRight(PixelPosition const &v0, PixelPosition const &v1,
                           PixelPosition const &v2) -> bool {
  return signed2DCross(v0, v1, v2) <= 0.0;
}

static void fanSort(std::vector<hpm::Mark> &fan) {
  const auto &pivot = fan[0];
  std::sort(std::next(fan.begin()), fan.end(),
            [&pivot](hpm::Mark const &lhs, hpm::Mark const &rhs) -> bool {
              return isRight(pivot.m_center, lhs.m_center, rhs.m_center);
            });
}

void sortCcw(std::vector<hpm::Mark> &marks) {
  if (not(isRight(marks[0].m_center, marks[1].m_center, marks[2].m_center))) {
    std::swap(marks[0], marks[1]);
  }
  fanSort(marks);
}

IdentifiedMarks::IdentifiedMarks(Marks const &marks, double const markerR,
                                 double const f,
                                 PixelPosition const &imageCenter) {
  if (marks.red.size() != 2 or marks.green.size() != 2 or
      marks.blue.size() != 2) {
    return;
  }

  std::vector<hpm::Mark> all{marks.getFlatCopy()};
  sortCcw(all);

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
