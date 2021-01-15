#pragma once

using Segment = std::vector<cv::Point>;

enum class EdgeDir { VERTICAL, HORIZONTAL, NONE };

int constexpr MAX_GRAD_VALUE{128 * 256};
double constexpr EPSILON{1.0};
size_t constexpr MIN_SEGMENT_LEN{10};
