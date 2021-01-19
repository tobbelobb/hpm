#pragma once

#define TABSIZE 100000

//----------------------------------------------
// Fast arctan2 using a lookup table
//
#define MAX_LUT_SIZE 1024

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif /* !M_LN10 */

/** PI */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif /* !M_PI */

#define RELATIVE_ERROR_FACTOR 100.0

// Lookup table (LUT) for NFA computation
class NFALUT {
public:
  NFALUT(int size, double _prob, double _logNT);
  ~NFALUT();

  int *LUT; // look up table
  int LUTSize;

  double prob;
  double logNT;

  auto checkValidationByNFA(int n, int k) -> bool;
  static auto myAtan2(double yy, double xx) -> double;

private:
  [[nodiscard]] auto nfa(int n, int k) const -> double;
  static auto log_gamma_lanczos(double x) -> double;
  static auto log_gamma_windschitl(double x) -> double;
  static auto log_gamma(double x) -> double;
  static auto double_equal(double a, double b) -> int;
};
