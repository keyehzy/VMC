#include "vmc/jastrow_parameterization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vmc {
namespace {

void require_site_count(std::size_t site_count) {
  if (site_count == 0) {
    throw std::invalid_argument("Jastrow parameterization must contain at least one site");
  }
}

void require_state_size(const BosonState& state, std::size_t site_count) {
  if (state.site_count() != site_count) {
    throw std::invalid_argument(
        "boson state and Jastrow parameterization must have the same site count");
  }
}

void require_finite_vector(const Eigen::VectorXd& values, const char* message) {
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) {
      throw std::invalid_argument(message);
    }
  }
}

void require_valid_matrix(const Eigen::MatrixXd& matrix) {
  if (matrix.rows() == 0 || matrix.cols() == 0) {
    throw std::invalid_argument("full Jastrow matrix must contain at least one site");
  }

  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("full Jastrow matrix must be square");
  }

  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      if (!std::isfinite(matrix(row, col))) {
        throw std::invalid_argument("full Jastrow matrix entries must be finite");
      }
    }
  }

  if (!matrix.isApprox(matrix.transpose())) {
    throw std::invalid_argument("full Jastrow matrix must be symmetric");
  }
}

std::size_t full_parameter_count(std::size_t site_count) {
  constexpr std::size_t max_size = std::numeric_limits<std::size_t>::max();
  if (site_count > max_size - 1 || site_count > max_size / (site_count + 1)) {
    throw std::overflow_error("full Jastrow parameter count overflows");
  }
  return site_count * (site_count + 1) / 2;
}

void require_parameter_size(const Eigen::VectorXd& parameters, std::size_t expected_size,
                            const char* message) {
  if (parameters.size() != static_cast<Eigen::Index>(expected_size)) {
    throw std::invalid_argument(message);
  }
}

Eigen::MatrixXd unpack_full_matrix(std::size_t site_count, const Eigen::VectorXd& parameters) {
  require_parameter_size(parameters, full_parameter_count(site_count),
                         "full Jastrow parameter vector has the wrong size");
  require_finite_vector(parameters, "full Jastrow parameters must be finite");

  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(site_count),
                                                 static_cast<Eigen::Index>(site_count));
  Eigen::Index parameter = 0;
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = row; col < matrix.cols(); ++col) {
      matrix(row, col) = parameters[parameter];
      matrix(col, row) = parameters[parameter];
      ++parameter;
    }
  }

  return matrix;
}

std::size_t shell_count(const Eigen::MatrixXi& shell_indices) {
  if (shell_indices.rows() == 0 || shell_indices.cols() == 0) {
    throw std::invalid_argument("distance Jastrow shell indices must contain at least one site");
  }

  if (shell_indices.rows() != shell_indices.cols()) {
    throw std::invalid_argument("distance Jastrow shell indices must be square");
  }

  if ((shell_indices.array() < 0).any()) {
    throw std::invalid_argument("distance Jastrow shell indices must be nonnegative");
  }

  if (!shell_indices.isApprox(shell_indices.transpose())) {
    throw std::invalid_argument("distance Jastrow shell indices must be symmetric");
  }

  const int max_shell = shell_indices.maxCoeff();
  std::vector<bool> seen(static_cast<std::size_t>(max_shell) + 1, false);
  for (Eigen::Index row = 0; row < shell_indices.rows(); ++row) {
    for (Eigen::Index col = 0; col < shell_indices.cols(); ++col) {
      seen[static_cast<std::size_t>(shell_indices(row, col))] = true;
    }
  }

  if (!std::all_of(seen.begin(), seen.end(), [](bool value) { return value; })) {
    throw std::invalid_argument("distance Jastrow shell indices must be contiguous from zero");
  }

  return seen.size();
}

Eigen::MatrixXi periodic_chain_shell_indices(std::size_t site_count) {
  require_site_count(site_count);

  Eigen::MatrixXi shell_indices(static_cast<Eigen::Index>(site_count),
                                static_cast<Eigen::Index>(site_count));

  for (std::size_t row = 0; row < site_count; ++row) {
    for (std::size_t col = 0; col < site_count; ++col) {
      const std::size_t separation = row > col ? row - col : col - row;
      const std::size_t distance = std::min(separation, site_count - separation);
      shell_indices(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          static_cast<int>(distance);
    }
  }

  return shell_indices;
}

}  // namespace

FullMatrixJastrowParameterization::FullMatrixJastrowParameterization(std::size_t site_count)
    : matrix_{Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(site_count),
                                    static_cast<Eigen::Index>(site_count))} {
  require_site_count(site_count);
}

FullMatrixJastrowParameterization::FullMatrixJastrowParameterization(Eigen::MatrixXd matrix)
    : matrix_{std::move(matrix)} {
  require_valid_matrix(matrix_);
}

std::size_t FullMatrixJastrowParameterization::site_count() const {
  return static_cast<std::size_t>(matrix_.rows());
}

std::size_t FullMatrixJastrowParameterization::parameter_count() const {
  return full_parameter_count(site_count());
}

Eigen::VectorXd FullMatrixJastrowParameterization::parameters() const {
  Eigen::VectorXd values(static_cast<Eigen::Index>(parameter_count()));
  Eigen::Index parameter = 0;
  for (Eigen::Index row = 0; row < matrix_.rows(); ++row) {
    for (Eigen::Index col = row; col < matrix_.cols(); ++col) {
      values[parameter] = matrix_(row, col);
      ++parameter;
    }
  }
  return values;
}

void FullMatrixJastrowParameterization::set_parameters(const Eigen::VectorXd& parameters) {
  matrix_ = unpack_full_matrix(site_count(), parameters);
}

void FullMatrixJastrowParameterization::update_parameters(const Eigen::VectorXd& delta) {
  set_parameters(parameters() + delta);
}

Eigen::MatrixXd FullMatrixJastrowParameterization::matrix() const {
  return matrix_;
}

Eigen::VectorXd FullMatrixJastrowParameterization::log_derivatives(const BosonState& state) const {
  require_state_size(state, site_count());

  Eigen::VectorXd derivatives(static_cast<Eigen::Index>(parameter_count()));
  Eigen::Index parameter = 0;
  for (BosonState::Site row = 0; row < state.site_count(); ++row) {
    const double row_occupation = static_cast<double>(state.occupation(row));
    for (BosonState::Site col = row; col < state.site_count(); ++col) {
      const double col_occupation = static_cast<double>(state.occupation(col));
      derivatives[parameter] =
          row == col ? -0.5 * row_occupation * col_occupation : -row_occupation * col_occupation;
      ++parameter;
    }
  }

  return derivatives;
}

DistanceJastrowParameterization::DistanceJastrowParameterization(Eigen::MatrixXi shell_indices,
                                                                 Eigen::VectorXd parameters)
    : shell_indices_{std::move(shell_indices)}, parameters_{std::move(parameters)} {
  const std::size_t expected_parameter_count = shell_count(shell_indices_);
  require_parameter_size(parameters_, expected_parameter_count,
                         "distance Jastrow parameter vector has the wrong size");
  require_finite_vector(parameters_, "distance Jastrow parameters must be finite");
}

DistanceJastrowParameterization DistanceJastrowParameterization::periodic_chain(
    std::size_t site_count) {
  return periodic_chain(site_count,
                        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(site_count / 2 + 1)));
}

DistanceJastrowParameterization DistanceJastrowParameterization::periodic_chain(
    std::size_t site_count, Eigen::VectorXd parameters) {
  return DistanceJastrowParameterization{periodic_chain_shell_indices(site_count),
                                         std::move(parameters)};
}

std::size_t DistanceJastrowParameterization::site_count() const {
  return static_cast<std::size_t>(shell_indices_.rows());
}

std::size_t DistanceJastrowParameterization::parameter_count() const {
  return static_cast<std::size_t>(parameters_.size());
}

Eigen::VectorXd DistanceJastrowParameterization::parameters() const {
  return parameters_;
}

void DistanceJastrowParameterization::set_parameters(const Eigen::VectorXd& parameters) {
  require_parameter_size(parameters, parameter_count(),
                         "distance Jastrow parameter vector has the wrong size");
  require_finite_vector(parameters, "distance Jastrow parameters must be finite");
  parameters_ = parameters;
}

void DistanceJastrowParameterization::update_parameters(const Eigen::VectorXd& delta) {
  set_parameters(parameters_ + delta);
}

Eigen::MatrixXd DistanceJastrowParameterization::matrix() const {
  Eigen::MatrixXd values(static_cast<Eigen::Index>(site_count()),
                         static_cast<Eigen::Index>(site_count()));
  for (Eigen::Index row = 0; row < values.rows(); ++row) {
    for (Eigen::Index col = 0; col < values.cols(); ++col) {
      values(row, col) = parameters_[shell_indices_(row, col)];
    }
  }
  return values;
}

Eigen::VectorXd DistanceJastrowParameterization::log_derivatives(const BosonState& state) const {
  require_state_size(state, site_count());

  Eigen::VectorXd derivatives = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(parameter_count()));
  for (BosonState::Site row = 0; row < state.site_count(); ++row) {
    const double row_occupation = static_cast<double>(state.occupation(row));
    for (BosonState::Site col = 0; col < state.site_count(); ++col) {
      const double col_occupation = static_cast<double>(state.occupation(col));
      const int shell =
          shell_indices_(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
      derivatives[shell] += -0.5 * row_occupation * col_occupation;
    }
  }

  return derivatives;
}

const Eigen::MatrixXi& DistanceJastrowParameterization::shell_indices() const {
  return shell_indices_;
}

}  // namespace vmc
