#pragma once

#include <vector>

// NFA: non-deterministic finite automata
// LUT: look up table

//----------------------------------------------
// Fast arctan2 using a lookup table
//

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif /* !M_LN10 */

// Lookup table (LUT) for NFA computation
class NFALUT {
public:
  NFALUT(int size, double _prob, double _logNT);

  std::vector<int> LUT{}; // look up table

  double prob;
  double logNT;

  bool checkValidationByNFA(int n, int k);
  static double myAtan2(double yy, double xx);

private:
  double nfa(int n, int k);
  static double log_gamma_lanczos(double x);
  static double log_gamma_windschitl(double x);
  static double log_gamma(double x);
  static bool double_equal(double a, double b);

  static constexpr size_t TABSIZE{1000};
  static constexpr size_t MAX_LUT_SIZE{1024};
};
