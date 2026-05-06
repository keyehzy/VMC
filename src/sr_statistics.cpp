#include "vmc/sr_statistics.hpp"

#include <cmath>
#include <stdexcept>

namespace vmc {
namespace {

void require_parameter_count(std::size_t parameter_count) {
  if (parameter_count == 0) {
    throw std::invalid_argument("SR accumulator requires at least one parameter");
  }
}

void require_finite_energy(double energy) {
  if (!std::isfinite(energy)) {
    throw std::invalid_argument("SR energy samples must be finite");
  }
}

void require_derivative_size(const Eigen::VectorXd& log_derivatives, std::size_t parameter_count) {
  if (log_derivatives.size() != static_cast<Eigen::Index>(parameter_count)) {
    throw std::invalid_argument("SR logarithmic derivative vector has the wrong size");
  }
}

void require_finite_derivatives(const Eigen::VectorXd& log_derivatives) {
  for (Eigen::Index i = 0; i < log_derivatives.size(); ++i) {
    if (!std::isfinite(log_derivatives[i])) {
      throw std::invalid_argument("SR logarithmic derivatives must be finite");
    }
  }
}

void require_measurement_inputs(const Lattice& lattice, const WaveFunction& wave_function,
                                const JastrowParameterization& parameterization,
                                const BosonState& state) {
  if (lattice.site_count() != state.site_count()) {
    throw std::invalid_argument("lattice and boson state must have the same site count");
  }

  if (wave_function.site_count() != state.site_count()) {
    throw std::invalid_argument("wave function and boson state must have the same site count");
  }

  if (parameterization.site_count() != state.site_count()) {
    throw std::invalid_argument(
        "Jastrow parameterization and boson state must have the same site count");
  }
}

Eigen::VectorXd zero_vector(std::size_t size) {
  return Eigen::VectorXd::Zero(static_cast<Eigen::Index>(size));
}

Eigen::MatrixXd zero_matrix(std::size_t size) {
  return Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(size), static_cast<Eigen::Index>(size));
}

}  // namespace

SrAccumulator::SrAccumulator(std::size_t parameter_count)
    : parameter_count_{parameter_count},
      derivative_sums_{zero_vector(parameter_count)},
      energy_derivative_sums_{zero_vector(parameter_count)},
      derivative_outer_sums_{zero_matrix(parameter_count)} {
  require_parameter_count(parameter_count_);
}

void SrAccumulator::add(double energy, const Eigen::VectorXd& log_derivatives) {
  require_finite_energy(energy);
  require_derivative_size(log_derivatives, parameter_count_);
  require_finite_derivatives(log_derivatives);

  ++sample_count_;
  energy_sum_ += energy;
  derivative_sums_ += log_derivatives;
  energy_derivative_sums_ += energy * log_derivatives;
  derivative_outer_sums_ += log_derivatives * log_derivatives.transpose();
}

std::size_t SrAccumulator::parameter_count() const {
  return parameter_count_;
}

std::size_t SrAccumulator::sample_count() const {
  return sample_count_;
}

double SrAccumulator::energy_mean() const {
  if (sample_count_ == 0) {
    return 0.0;
  }
  return energy_sum_ / static_cast<double>(sample_count_);
}

Eigen::VectorXd SrAccumulator::derivative_means() const {
  if (sample_count_ == 0) {
    return zero_vector(parameter_count_);
  }
  return derivative_sums_ / static_cast<double>(sample_count_);
}

Eigen::VectorXd SrAccumulator::energy_derivative_means() const {
  if (sample_count_ == 0) {
    return zero_vector(parameter_count_);
  }
  return energy_derivative_sums_ / static_cast<double>(sample_count_);
}

Eigen::MatrixXd SrAccumulator::derivative_outer_means() const {
  if (sample_count_ == 0) {
    return zero_matrix(parameter_count_);
  }
  return derivative_outer_sums_ / static_cast<double>(sample_count_);
}

Eigen::MatrixXd SrAccumulator::derivative_covariance() const {
  const Eigen::VectorXd means = derivative_means();
  return derivative_outer_means() - means * means.transpose();
}

Eigen::VectorXd SrAccumulator::forces() const {
  return -2.0 * (energy_derivative_means() - energy_mean() * derivative_means());
}

SrMeasurementResult measure_sr_statistics(const Lattice& lattice, const WaveFunction& wave_function,
                                          const JastrowParameterization& parameterization,
                                          const BoseHubbardHamiltonian& hamiltonian,
                                          BosonState& state, std::size_t thermalization_sweeps,
                                          std::size_t sample_count,
                                          std::size_t sweeps_between_samples,
                                          std::mt19937_64& rng) {
  require_measurement_inputs(lattice, wave_function, parameterization, state);
  if (sample_count > 0 && sweeps_between_samples == 0) {
    throw std::invalid_argument("SR measurement requires at least one sweep between samples");
  }

  SrMeasurementResult result{
      .thermalization_stats =
          run_metropolis_sweeps(lattice, wave_function, state, thermalization_sweeps, rng),
      .sampling_stats =
          MetropolisRunStats{
              .attempted_steps = 0,
              .proposed_steps = 0,
              .accepted_steps = 0,
          },
      .statistics = SrAccumulator{parameterization.parameter_count()},
  };

  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    result.sampling_stats +=
        run_metropolis_sweeps(lattice, wave_function, state, sweeps_between_samples, rng);
    result.statistics.add(local_energy(lattice, state, wave_function, hamiltonian),
                          parameterization.log_derivatives(state));
  }

  return result;
}

}  // namespace vmc
