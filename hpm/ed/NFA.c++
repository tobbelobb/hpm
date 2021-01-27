#include <cfloat>
#include <cmath>
#include <limits>

#include <hpm/ed/NFA.h++>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wconversion"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Walloc-size-larger-than="
#endif

NFALUT::NFALUT(int size, double _prob, double _logNT) {
  LUTSize = size;
  LUT = new int[LUTSize];

  prob = _prob;
  logNT = _logNT;

  LUT[0] = 1;
  int j = 1;
  for (int i = 1; i < LUTSize; i++) {
    LUT[i] = LUTSize + 1;
    double ret = nfa(i, j);
    if (ret < 0) {
      while (j < i) {
        j++;
        ret = nfa(i, j);
        if (ret >= 0) {
          break;
        }
      }

      if (ret < 0) {
        continue;
      }
    }

    LUT[i] = j;
  }
}

NFALUT::~NFALUT() { delete[] LUT; }

auto NFALUT::checkValidationByNFA(int n, int k) -> bool {
  if (n >= LUTSize) {
    return nfa(n, k) >= 0.0;
  }
  { return k >= LUT[n]; }
}

auto NFALUT::myAtan2(double yy, double xx) -> double {
  static double LUT[MAX_LUT_SIZE + 1];
  static bool tableInited = false;
  if (!tableInited) {
    for (int i = 0; i <= MAX_LUT_SIZE; i++) {
      LUT[i] = atan(static_cast<double>(i) / MAX_LUT_SIZE);
    }

    tableInited = true;
  }

  double y = fabs(yy);
  double x = fabs(xx);

  bool invert = false;
  if (y > x) {
    double t = x;
    x = y;
    y = t;
    invert = true;
  }

  double ratio = std::numeric_limits<double>::quiet_NaN();
  if (x == 0) { // avoid division error
    x = 0.000001;
  }

  ratio = y / x;

  double angle = LUT[static_cast<int>(ratio * MAX_LUT_SIZE)];

  if (xx >= 0) {
    if (yy >= 0) {
      // I. quadrant
      if (invert) {
        angle = M_PI / 2 - angle;
      }

    } else {
      // IV. quadrant
      if (!invert) {
        angle = M_PI - angle;
      } else {
        angle = M_PI / 2 + angle;
      }
    }

  } else {
    if (yy >= 0) {
      /// II. quadrant
      if (!invert) {
        angle = M_PI - angle;
      } else {
        angle = M_PI / 2 + angle;
      }

    } else {
      /// III. quadrant
      if (invert) {
        angle = M_PI / 2 - angle;
      }
    }
  }

  return angle;
}

auto NFALUT::nfa(int n, int k) const -> double {
  static double inv[TABSIZE]; /* table to keep computed inverse values */
  double tolerance = 0.1;     /* an error of 10% in the result is accepted */
  double log1term = std::numeric_limits<double>::quiet_NaN();
  double term = std::numeric_limits<double>::quiet_NaN();
  double bin_term = std::numeric_limits<double>::quiet_NaN();
  double mult_term = std::numeric_limits<double>::quiet_NaN();
  double bin_tail = std::numeric_limits<double>::quiet_NaN();
  double err = std::numeric_limits<double>::quiet_NaN();
  double p_term = std::numeric_limits<double>::quiet_NaN();
  int i = 0;

  /* check parameters */
  if (n < 0 || k < 0 || k > n || prob <= 0.0 || prob >= 1.0) {
    return -1.0;
  }

  /* trivial cases */
  if (n == 0 || k == 0) {
    return -logNT;
  }
  if (n == k) {
    return -logNT - static_cast<double>(n) * log10(prob);
  }

  /* probability term */
  p_term = prob / (1.0 - prob);

  /* compute the first term of the series */
  /*
  binomial_tail(n,k,p) = sum_{i=k}^n bincoef(n,i) * p^i * (1-p)^{n-i}
  where bincoef(n,i) are the binomial coefficients.
  But
  bincoef(n,k) = gamma(n+1) / ( gamma(k+1) * gamma(n-k+1) ).
  We use this to compute the first term. Actually the log of it.
  */
  log1term = log_gamma(static_cast<double>(n) + 1.0) -
             log_gamma(static_cast<double>(k) + 1.0) -
             log_gamma(static_cast<double>(n - k) + 1.0) +
             static_cast<double>(k) * log(prob) +
             static_cast<double>(n - k) * log(1.0 - prob);
  term = exp(log1term);

  /* in some cases no more computations are needed */
  if (double_equal(term, 0.0) != 0) { /* the first term is almost zero */
    if (static_cast<double>(k) >
        static_cast<double>(n) * prob) { /* at begin or end of the tail?  */
      return -log1term / M_LN10 - logNT; /* end: use just the first term  */
    }
    { return -logNT; /* begin: the tail is roughly 1  */ }
  }

  /* compute more terms if needed */
  bin_tail = term;
  for (i = k + 1; i <= n; i++) {
    /*
    As
    term_i = bincoef(n,i) * p^i * (1-p)^(n-i)
    and
    bincoef(n,i)/bincoef(n,i-1) = n-1+1 / i,
    then,
    term_i / term_i-1 = (n-i+1)/i * p/(1-p)
    and
    term_i = term_i-1 * (n-i+1)/i * p/(1-p).
    1/i is stored in a table as they are computed,
    because divisions are expensive.
    p/(1-p) is computed only once and stored in 'p_term'.
    */
    bin_term =
        static_cast<double>(n - i + 1) *
        (i < TABSIZE ? (inv[i] != 0.0 ? inv[i]
                                      : (inv[i] = 1.0 / static_cast<double>(i)))
                     : 1.0 / static_cast<double>(i));

    mult_term = bin_term * p_term;
    term *= mult_term;
    bin_tail += term;

    if (bin_term < 1.0) {
      /* When bin_term<1 then mult_term_j<mult_term_i for j>i.
      Then, the error on the binomial tail when truncated at
      the i term can be bounded by a geometric series of form
      term_i * sum mult_term_i^j.                            */
      err = term * ((1.0 - pow(mult_term, static_cast<double>(n - i + 1))) /
                        (1.0 - mult_term) -
                    1.0);

      /* One wants an error at most of tolerance*final_result, or:
      tolerance * abs(-log10(bin_tail)-logNT).
      Now, the error that can be accepted on bin_tail is
      given by tolerance*final_result divided by the derivative
      of -log10(x) when x=bin_tail. that is:
      tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
      Finally, we truncate the tail if the error is less than:
      tolerance * abs(-log10(bin_tail)-logNT) * bin_tail        */
      if (err < tolerance * fabs(-log10(bin_tail) - logNT) * bin_tail) {
        break;
      }
    }
  }

  return -log10(bin_tail) - logNT;
}

auto NFALUT::log_gamma_lanczos(double x) -> double {
  static double q[7] = {75122.6331530, 80916.6278952, 36308.2951477,
                        8687.24529705, 1168.92649479, 83.8676043424,
                        2.50662827511};
  double a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
  double b = 0.0;
  int n = 0;

  for (n = 0; n < 7; n++) {
    a -= log(x + static_cast<double>(n));
    b += q[n] * pow(x, static_cast<double>(n));
  }
  return a + log(b);
}

auto NFALUT::log_gamma_windschitl(double x) -> double {
  return 0.918938533204673 + (x - 0.5) * log(x) - x +
         0.5 * x * log(x * sinh(1 / x) + 1 / (810.0 * pow(x, 6.0)));
}

auto NFALUT::log_gamma(double x) -> double {
  return x > 15 ? log_gamma_windschitl(x) : log_gamma_lanczos(x);
}

auto NFALUT::double_equal(double a, double b) -> int {
  if (a == b) {
    return TRUE;
  }

  double const abs_diff = fabs(a - b);
  double const aa = fabs(a);
  double const bb = fabs(b);
  double abs_max = aa > bb ? aa : bb;

  /* DBL_MIN is the smallest normalized number, thus, the smallest
  number whose relative error is bounded by DBL_EPSILON. For
  smaller numbers, the same quantization steps as for DBL_MIN
  are used. Then, for smaller numbers, a meaningful "relative"
  error should be computed by dividing the difference by DBL_MIN. */
  if (abs_max < DBL_MIN) {
    abs_max = DBL_MIN;
  }

  /* equal if relative error <= factor x eps */
  return static_cast<int>((abs_diff / abs_max) <=
                          (RELATIVE_ERROR_FACTOR * DBL_EPSILON));
}
#pragma GCC diagnostic pop
