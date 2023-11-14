#pragma once

// clang-format off
#if defined(__clang__)
  #define DISABLE_WARNINGS \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wold-style-cast\"") \
    _Pragma("GCC diagnostic ignored \"-Wconversion\"") \
    _Pragma("GCC diagnostic ignored \"-Wsign-conversion\"") \
    _Pragma("GCC diagnostic ignored \"-Wc11-extensions\"") \
    _Pragma("GCC diagnostic ignored \"-Wdeprecated-anon-enum-enum-conversion\"") \
    _Pragma("GCC diagnostic ignored \"-Wcast-align\"") \
    _Pragma("GCC diagnostic ignored \"-Wdouble-promotion\"") \
    _Pragma("GCC diagnostic ignored \"-Wpedantic\"")
#elif defined(__GNUC__)
  #define DISABLE_WARNINGS \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wold-style-cast\"") \
    _Pragma("GCC diagnostic ignored \"-Wconversion\"") \
    _Pragma("GCC diagnostic ignored \"-Wsign-conversion\"") \
    _Pragma("GCC diagnostic ignored \"-Wdouble-promotion\"") \
    _Pragma("GCC diagnostic ignored \"-Wfloat-conversion\"") \
    _Pragma("GCC diagnostic ignored \"-Woverloaded-virtual\"") \
    _Pragma("GCC diagnostic ignored \"-Wpedantic\"")
    //_Pragma("GCC diagnostic ignored \"-Wdeprecated-enum-enum-conversion\"")
#elif defined(_MSC_VER)
  #define DISABLE_WARNINGS \
    __pragma("warning(push, 0)")
#else
  #define DISABLE_WARNINGS
#endif

#if defined(__clang__) || defined(__GNUC__)
  #define ENABLE_WARNINGS \
    _Pragma("GCC diagnostic pop")
#elif defined(_MSC_VER)
  #define ENABLE_WARNINGS \
    __pragma("warning(pop)")
#else
  #define ENABLE_WARNINGS
#endif
    // clang-format on
