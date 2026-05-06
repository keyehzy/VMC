#pragma once

#include <cstddef>

namespace vmc {

class RunningStats {
 public:
  void add(double value);

  [[nodiscard]] std::size_t count() const;
  [[nodiscard]] double mean() const;
  [[nodiscard]] double variance() const;
  [[nodiscard]] double standard_deviation() const;
  [[nodiscard]] double standard_error() const;

 private:
  std::size_t count_{0};
  double mean_{0.0};
  double m2_{0.0};
};

class BinningStats {
 public:
  explicit BinningStats(std::size_t bin_size);

  void add(double value);

  [[nodiscard]] std::size_t bin_size() const;
  [[nodiscard]] std::size_t sample_count() const;
  [[nodiscard]] std::size_t completed_bin_count() const;
  [[nodiscard]] std::size_t current_bin_count() const;

  [[nodiscard]] const RunningStats& raw_stats() const;
  [[nodiscard]] const RunningStats& bin_stats() const;

 private:
  std::size_t bin_size_;
  std::size_t sample_count_{0};
  std::size_t current_bin_count_{0};
  double current_bin_sum_{0.0};
  RunningStats raw_stats_;
  RunningStats bin_stats_;
};

}  // namespace vmc
