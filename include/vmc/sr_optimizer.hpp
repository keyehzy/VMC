#pragma once

#include <Eigen/Dense>

#include "vmc/sr_statistics.hpp"

namespace vmc {

struct SrUpdateConfig {
  double step_size;
  double diagonal_shift;
  double singular_tolerance{1.0e-12};
};

struct SrUpdateResult {
  Eigen::VectorXd delta;
  Eigen::MatrixXd covariance;
  Eigen::MatrixXd regularized_covariance;
  Eigen::VectorXd forces;
};

[[nodiscard]] SrUpdateResult compute_sr_update(const SrAccumulator& statistics,
                                               const SrUpdateConfig& config);

}  // namespace vmc
