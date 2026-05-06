#pragma once

#include <Eigen/Dense>
#include <cstddef>

#include "vmc/boson_state.hpp"
#include "vmc/proposal.hpp"

namespace vmc {

class CondensateWaveFunction {
 public:
  explicit CondensateWaveFunction(Eigen::VectorXd orbital);

  static CondensateWaveFunction uniform(std::size_t site_count);

  [[nodiscard]] std::size_t site_count() const;

  [[nodiscard]] double log_amplitude(const BosonState& state) const;
  [[nodiscard]] double log_ratio(const BosonState& state, const BosonHop& hop) const;
  [[nodiscard]] double ratio(const BosonState& state, const BosonHop& hop) const;

 private:
  Eigen::VectorXd orbital_;
};

}  // namespace vmc
