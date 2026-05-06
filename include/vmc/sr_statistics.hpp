#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <random>

#include "vmc/boson_state.hpp"
#include "vmc/hamiltonian.hpp"
#include "vmc/jastrow_parameterization.hpp"
#include "vmc/lattice.hpp"
#include "vmc/metropolis.hpp"
#include "vmc/wave_function.hpp"

namespace vmc {

class SrAccumulator {
 public:
  explicit SrAccumulator(std::size_t parameter_count);

  void add(double energy, const Eigen::VectorXd& log_derivatives);

  [[nodiscard]] std::size_t parameter_count() const;
  [[nodiscard]] std::size_t sample_count() const;

  [[nodiscard]] double energy_mean() const;
  [[nodiscard]] Eigen::VectorXd derivative_means() const;
  [[nodiscard]] Eigen::VectorXd energy_derivative_means() const;
  [[nodiscard]] Eigen::MatrixXd derivative_outer_means() const;
  [[nodiscard]] Eigen::MatrixXd derivative_covariance() const;
  [[nodiscard]] Eigen::VectorXd forces() const;

 private:
  std::size_t parameter_count_;
  std::size_t sample_count_{0};
  double energy_sum_{0.0};
  Eigen::VectorXd derivative_sums_;
  Eigen::VectorXd energy_derivative_sums_;
  Eigen::MatrixXd derivative_outer_sums_;
};

struct SrMeasurementResult {
  MetropolisRunStats thermalization_stats;
  MetropolisRunStats sampling_stats;
  SrAccumulator statistics;
};

SrMeasurementResult measure_sr_statistics(const Lattice& lattice, const WaveFunction& wave_function,
                                          const JastrowParameterization& parameterization,
                                          const BoseHubbardHamiltonian& hamiltonian,
                                          BosonState& state, std::size_t thermalization_sweeps,
                                          std::size_t sample_count,
                                          std::size_t sweeps_between_samples, std::mt19937_64& rng);

}  // namespace vmc
