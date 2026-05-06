#include "vmc/sr_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vmc {
namespace {

void require_samples(const SrAccumulator& statistics) {
  if (statistics.sample_count() == 0) {
    throw std::invalid_argument("SR updates require at least one sample");
  }
}

void require_config(const SrUpdateConfig& config) {
  if (!std::isfinite(config.step_size) || config.step_size <= 0.0) {
    throw std::invalid_argument("SR update step size must be finite and positive");
  }

  if (!std::isfinite(config.diagonal_shift) || config.diagonal_shift < 0.0) {
    throw std::invalid_argument("SR update diagonal shift must be finite and nonnegative");
  }

  if (!std::isfinite(config.singular_tolerance) || config.singular_tolerance < 0.0) {
    throw std::invalid_argument("SR update singular tolerance must be finite and nonnegative");
  }
}

void require_finite_vector(const Eigen::VectorXd& values, const char* message) {
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) {
      throw std::invalid_argument(message);
    }
  }
}

void require_finite_matrix(const Eigen::MatrixXd& values, const char* message) {
  for (Eigen::Index row = 0; row < values.rows(); ++row) {
    for (Eigen::Index col = 0; col < values.cols(); ++col) {
      if (!std::isfinite(values(row, col))) {
        throw std::invalid_argument(message);
      }
    }
  }
}

void require_symmetric_matrix(const Eigen::MatrixXd& values) {
  if (values.rows() != values.cols()) {
    throw std::invalid_argument("SR covariance matrix must be square");
  }

  if (!values.isApprox(values.transpose())) {
    throw std::invalid_argument("SR covariance matrix must be symmetric");
  }
}

void require_well_conditioned(const Eigen::MatrixXd& matrix, double singular_tolerance) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver{matrix};
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("SR covariance eigensolve failed");
  }

  const double min_eigenvalue = eigensolver.eigenvalues().minCoeff();
  const double max_abs_eigenvalue = eigensolver.eigenvalues().cwiseAbs().maxCoeff();
  const double scale = std::max(1.0, max_abs_eigenvalue);
  if (min_eigenvalue <= singular_tolerance * scale) {
    throw std::runtime_error("SR regularized covariance matrix is singular");
  }
}

Eigen::MatrixXd regularized_covariance(const Eigen::MatrixXd& covariance, double diagonal_shift) {
  Eigen::MatrixXd regularized = covariance;
  regularized.diagonal().array() += diagonal_shift;
  return regularized;
}

}  // namespace

SrUpdateResult compute_sr_update(const SrAccumulator& statistics, const SrUpdateConfig& config) {
  require_samples(statistics);
  require_config(config);

  const Eigen::MatrixXd covariance = statistics.derivative_covariance();
  const Eigen::VectorXd forces = statistics.forces();
  require_symmetric_matrix(covariance);
  require_finite_matrix(covariance, "SR covariance matrix must be finite");
  require_finite_vector(forces, "SR forces must be finite");

  const Eigen::MatrixXd regularized = regularized_covariance(covariance, config.diagonal_shift);
  require_well_conditioned(regularized, config.singular_tolerance);

  Eigen::LDLT<Eigen::MatrixXd> decomposition{regularized};
  if (decomposition.info() != Eigen::Success) {
    throw std::runtime_error("SR covariance factorization failed");
  }

  const Eigen::VectorXd delta = decomposition.solve(config.step_size * forces);
  require_finite_vector(delta, "SR update delta must be finite");

  return SrUpdateResult{
      .delta = delta,
      .covariance = covariance,
      .regularized_covariance = regularized,
      .forces = forces,
  };
}

}  // namespace vmc
