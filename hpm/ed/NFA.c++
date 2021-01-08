#include <array>
#include <cmath>
#include <limits>

#include <hpm/ed/NFA.h++>

NFALUT::NFALUT(size_t size, double _prob, double _logNT)
    : prob(_prob), logNT(_logNT) {
  LUT.reserve(size);

  LUT.push_back(1);
  size_t j = 1;
  for (size_t i = 1; i < size; ++i) {
    LUT.push_back(size + 1);
    double ret = nfa(i, j);
    if (ret < 0) {
      while (j < i and ret < 0) {
        j++;
        ret = nfa(i, j);
      }
    }

    if (ret >= 0) {
      LUT.back() = j;
    }
  }
}

bool NFALUT::checkValidationByNFA(size_t n, size_t k) {
  if (n >= LUT.size()) {
    return nfa(n, k) >= 0.0;
  }
  return k >= LUT[n];
}

double NFALUT::myAtan2(double yy, double xx) {
  static std::array<double, MAX_LUT_SIZE + 1> atan2LUT{};
  static bool tableInited = false;
  if (not tableInited) {
    for (size_t i{0}; i <= MAX_LUT_SIZE; i++) {
      atan2LUT[i] =
          atan(static_cast<double>(i) / static_cast<double>(MAX_LUT_SIZE));
    }
    tableInited = true;
  }

  double y = std::abs(yy);
  double x = std::abs(xx);

  bool invert = false;
  if (y > x) {
    std::swap(x, y);
    invert = true;
  }

  if (x == 0) {
    x = 0.000001;
  }

  double const ratio = y / x;

  double angle = atan2LUT[static_cast<size_t>(ratio * MAX_LUT_SIZE)];

  if (xx >= 0.0) {
    if (yy >= 0.0) {
      if (invert) {
        angle = M_PI / 2 - angle;
      }
    } else {
      if (invert) {
        angle = M_PI / 2 + angle;
      } else {
        angle = M_PI - angle;
      }
    }
  } else {
    if (yy >= 0.0) {
      if (invert) {
        angle = M_PI / 2 + angle;
      } else {
        angle = M_PI - angle;
      }
    } else {
      if (invert) {
        angle = M_PI / 2 - angle;
      }
    }
  }

  return angle;
}

double NFALUT::nfa(size_t n, size_t k) {
  double constexpr LN10{2.30258509299404568402};
  static std::array<double, TABSIZE> inv{};
  double const nd{static_cast<double>(n)};
  double const kd{static_cast<double>(k)};

  if (k > n or prob <= 0.0 or prob >= 1.0) {
    return -1.0;
  }

  if (n == 0 || k == 0) {
    return -logNT;
  }
  if (n == k) {
    return -logNT - nd * log10(prob);
  }

  /* compute the first term of the series */
  /*
  binomial_tail(n,k,p) = sum_{i=k}^n bincoef(n,i) * p^i * (1-p)^{n-i}
  where bincoef(n,i) are the binomial coefficients.
  But
  bincoef(n,k) = gamma(n+1) / ( gamma(k+1) * gamma(n-k+1) ).
  We use this to compute the first term. Actually the log of it.
  */
  double const log1term = log_gamma(nd + 1.0) - log_gamma(kd + 1.0) -
                          log_gamma((nd - kd) + 1.0) + kd * log(prob) +
                          (nd - kd) * log(1.0 - prob);
  double term = exp(log1term);

  /* in some cases no more computations are needed */
  if (double_equal(term, 0.0)) {
    if (kd > nd * prob) {              /* at begin or end of the tail?  */
      return -log1term / LN10 - logNT; /* end: use just the first term  */
    } else {
      return -logNT; /* begin: the tail is roughly 1  */
    }
  }

  double bin_tail = term;
  for (size_t i = k + 1; i <= n; i++) {
    double const id{static_cast<double>(i)};
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
    p/(1-p) is computed only once and stored in 'probability_term'.
    */
    double const bin_term =
        (nd - id + 1.0) * (i < TABSIZE
                               ? (inv[i] != 0.0 ? inv[i] : (inv[i] = 1.0 / id))
                               : 1.0 / id);

    double const mult_term = bin_term * (prob / (1.0 - prob));
    term *= mult_term;
    bin_tail += term;

    if (bin_term < 1.0) {
      /* When bin_term<1 then mult_term_j<mult_term_i for j>i.
      Then, the error on the binomial tail when truncated at
      the i term can be bounded by a geometric series of form
      term_i * sum mult_term_i^j.                            */
      double const err =
          term *
          ((1.0 - pow(mult_term, (nd - id + 1.0))) / (1.0 - mult_term) - 1.0);

      /* One wants an error at most of tolerance*final_result, or:
      tolerance * abs(-log10(bin_tail)-logNT).
      Now, the error that can be accepted on bin_tail is
      given by tolerance*final_result divided by the derivative
      of -log10(x) when x=bin_tail. that is:
      tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
      Finally, we truncate the tail if the error is less than:
      tolerance * abs(-log10(bin_tail)-logNT) * bin_tail        */
      double constexpr TOLERANCE{0.1};
      if (err < TOLERANCE * std::abs(-log10(bin_tail) - logNT) * bin_tail) {
        break;
      }
    }
  }

  return -log10(bin_tail) - logNT;
}

double NFALUT::log_gamma_lanczos(double x) {
  static double q[7] = {75122.6331530, 80916.6278952, 36308.2951477,
                        8687.24529705, 1168.92649479, 83.8676043424,
                        2.50662827511};
  double a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
  double b = 0.0;

  for (size_t n = 0; n < 7; n++) {
    double const nd{static_cast<double>(n)};
    a -= log(x + nd);
    b += q[n] * pow(x, nd);
  }
  return a + log(b);
}

double NFALUT::log_gamma_windschitl(double x) {
  return 0.918938533204673 + (x - 0.5) * log(x) - x +
         0.5 * x * log(x * sinh(1 / x) + 1 / (810.0 * pow(x, 6.0)));
}

double NFALUT::log_gamma(double x) {
  return x > 15.0 ? log_gamma_windschitl(x) : log_gamma_lanczos(x);
}

bool NFALUT::double_equal(double a, double b) {
  if (a == b) {
    return true;
  }

  double const abs_diff = std::abs(a - b);

  double const aa = std::abs(a);
  double const bb = std::abs(b);
  double constexpr MIN{std::numeric_limits<double>::min()};
  double const abs_max = std::max(std::max(aa, bb), MIN);

  double constexpr RELATIVE_ERROR_FACTOR{100.0};
  double constexpr EPSILON{std::numeric_limits<double>::epsilon()};
  return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * EPSILON);
}
