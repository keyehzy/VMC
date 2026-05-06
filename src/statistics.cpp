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

BinningStats::BinningStats(std::size_t bin_size) : bin_size_{bin_size} {
  if (bin_size_ == 0) {
    throw std::invalid_argument("bin size must be greater than zero");
  }
}

void BinningStats::add(double value) {
  raw_stats_.add(value);

  ++sample_count_;
  ++current_bin_count_;
  current_bin_sum_ += value;

  if (current_bin_count_ == bin_size_) {
    bin_stats_.add(current_bin_sum_ / static_cast<double>(bin_size_));
    current_bin_count_ = 0;
    current_bin_sum_ = 0.0;
  }
}

std::size_t BinningStats::bin_size() const {
  return bin_size_;
}

std::size_t BinningStats::sample_count() const {
  return sample_count_;
}

std::size_t BinningStats::completed_bin_count() const {
  return bin_stats_.count();
}

std::size_t BinningStats::current_bin_count() const {
  return current_bin_count_;
}

const RunningStats& BinningStats::raw_stats() const {
  return raw_stats_;
}

const RunningStats& BinningStats::bin_stats() const {
  return bin_stats_;
}

}  // namespace vmc
