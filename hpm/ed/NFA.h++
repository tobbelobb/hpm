#pragma once

#include <vector>

// Fast arctan2 using a lookup table

// NFA: non-deterministic finite automata
// LUT: look up table

// Lookup table (LUT) for NFA computation
class NFALUT {
public:
  NFALUT(size_t size, double _prob, double _logNT);

  std::vector<size_t> LUT{}; // look up table

  double prob;
  double logNT;

  bool checkValidationByNFA(size_t n, size_t k);
  static double myAtan2(double yy, double xx);

private:
  double nfa(size_t n, size_t k);
  static double log_gamma_lanczos(double x);
  static double log_gamma_windschitl(double x);
  static double log_gamma(double x);
  static bool double_equal(double a, double b);

  static constexpr size_t TABSIZE{1000};
  static constexpr size_t MAX_LUT_SIZE{1024};
};
