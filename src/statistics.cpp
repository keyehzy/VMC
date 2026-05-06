#include "vmc/statistics.hpp"

#include <cmath>
#include <stdexcept>

namespace vmc {

void RunningStats::add(double value) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument("running statistics values must be finite");
  }

  ++count_;
  const double delta = value - mean_;
  mean_ += delta / static_cast<double>(count_);
  const double delta2 = value - mean_;
  m2_ += delta * delta2;
}

std::size_t RunningStats::count() const {
  return count_;
}

double RunningStats::mean() const {
  return mean_;
}

double RunningStats::variance() const {
  if (count_ < 2) {
    return 0.0;
  }
  return m2_ / static_cast<double>(count_ - 1);
}

double RunningStats::standard_deviation() const {
  return std::sqrt(variance());
}

double RunningStats::standard_error() const {
  if (count_ < 2) {
    return 0.0;
  }
  return standard_deviation() / std::sqrt(static_cast<double>(count_));
}

}  // namespace vmc
