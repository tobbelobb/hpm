#include <hpm/ed/EDCommon.h++>

auto NFA(double prob, int len, int const numberOfSegmentPieces) -> double {
  auto nfa = static_cast<double>(numberOfSegmentPieces);
  for (int i = 0; i < len && nfa > EPSILON; i++) {
    nfa *= prob;
  }
  return nfa;
}

//----------------------------------------------------------------------------------------------
// After the validation of the edge segments, extracts the valid ones
// In other words, updates the valid segments' pixel arrays and their lengths
auto validSegments(cv::Mat edgeImageIn, std::vector<Segment> const &segmentsIn)
    -> std::vector<Segment> {
  std::vector<Segment> valids;
  int const width{edgeImageIn.cols};

  auto *edgeImg = edgeImageIn.ptr<uint8_t>(0);
  for (auto const &segment : segmentsIn) {
    auto const end = segment.end();
    auto front = segment.begin();
    auto back = segment.begin();
    while (back != end) {
      front = std::find_if(front, end, [&](auto const &point) {
        return edgeImg[point.y * width + point.x];
      });
      back = std::find_if_not(front, end, [&](auto const &point) {
        return edgeImg[point.y * width + point.x];
      });

      if (std::distance(front, back) >= static_cast<ssize_t>(MIN_SEGMENT_LEN)) {
        valids.emplace_back(front, std::prev(back));
      }
      front = std::next(back);
    }
  }

  return valids;
}
