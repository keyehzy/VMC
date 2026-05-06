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

}  // namespace vmc
