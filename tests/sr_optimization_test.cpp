#include "vmc/sr_optimization.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using testing::ElementsAre;
using vmc::BoseHubbardHamiltonian;
using vmc::BosonState;
using vmc::BoundaryCondition;
using vmc::CondensateWaveFunction;
using vmc::FullMatrixJastrowParameterization;
using vmc::Lattice;
using vmc::OccupancyConstraint;
using vmc::SrIterationConfig;
using vmc::SrOptimizationConfig;
using vmc::SrOptimizationResult;
using vmc::SrUpdateConfig;

std::vector<double> values_of(const Eigen::VectorXd& values) {
  return {values.data(), values.data() + values.size()};
}

MATCHER_P(DoubleNear, expected, "") {
  return testing::ExplainMatchResult(testing::DoubleNear(expected, 1.0e-12), arg, result_listener);
}

SrIterationConfig iteration_config() {
  return SrIterationConfig{
      .thermalization_sweeps = 0,
      .sample_count = 1,
      .sweeps_between_samples = 1,
      .update =
          SrUpdateConfig{
              .step_size = 0.25,
              .diagonal_shift = 1.0,
          },
  };
}

TEST(SrOptimizationTest, StopsWhenEnergyConverges) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(2);
  FullMatrixJastrowParameterization parameterization{2};
  Eigen::VectorXd initial_parameters(3);
  initial_parameters << 0.25, -0.5, 1.0;
  parameterization.set_parameters(initial_parameters);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 0.0,
      .interaction_u = 0.0,
  };
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  const SrOptimizationResult result =
      run_sr_optimization(lattice, condensate, parameterization, hamiltonian, state,
                          SrOptimizationConfig{
                              .max_iterations = 5,
                              .energy_tolerance = 0.0,
                              .iteration = iteration_config(),
                          },
                          rng);

  ASSERT_EQ(result.history.size(), 2);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.history[0].iteration, 0);
  EXPECT_EQ(result.history[1].iteration, 1);
  EXPECT_DOUBLE_EQ(result.history[0].energy_mean, 0.0);
  EXPECT_DOUBLE_EQ(result.history[1].energy_mean, 0.0);
  EXPECT_TRUE(result.history[0].delta.isZero());
  EXPECT_TRUE(result.history[1].delta.isZero());
  EXPECT_THAT(values_of(result.history[0].parameters_before),
              ElementsAre(DoubleNear(0.25), DoubleNear(-0.5), DoubleNear(1.0)));
  EXPECT_THAT(values_of(result.history[1].parameters_after),
              ElementsAre(DoubleNear(0.25), DoubleNear(-0.5), DoubleNear(1.0)));
  EXPECT_THAT(values_of(parameterization.parameters()),
              ElementsAre(DoubleNear(0.25), DoubleNear(-0.5), DoubleNear(1.0)));
}

TEST(SrOptimizationTest, StopsAtMaximumIterationsWhenNotEnoughHistoryToConverge) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(2);
  FullMatrixJastrowParameterization parameterization{2};
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 0.0,
      .interaction_u = 0.0,
  };
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  const SrOptimizationResult result =
      run_sr_optimization(lattice, condensate, parameterization, hamiltonian, state,
                          SrOptimizationConfig{
                              .max_iterations = 1,
                              .energy_tolerance = 0.0,
                              .iteration = iteration_config(),
                          },
                          rng);

  EXPECT_FALSE(result.converged);
  ASSERT_EQ(result.history.size(), 1);
  EXPECT_EQ(result.history[0].iteration, 0);
  EXPECT_DOUBLE_EQ(result.history[0].energy_mean, 0.0);
}

TEST(SrOptimizationTest, RejectsInvalidOptimizationConfig) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(2);
  FullMatrixJastrowParameterization parameterization{2};
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 0.0,
      .interaction_u = 0.0,
  };
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  EXPECT_THROW((void)run_sr_optimization(lattice, condensate, parameterization, hamiltonian, state,
                                         SrOptimizationConfig{
                                             .max_iterations = 0,
                                             .energy_tolerance = 0.0,
                                             .iteration = iteration_config(),
                                         },
                                         rng),
               std::invalid_argument);
  EXPECT_THROW((void)run_sr_optimization(lattice, condensate, parameterization, hamiltonian, state,
                                         SrOptimizationConfig{
                                             .max_iterations = 1,
                                             .energy_tolerance = -1.0,
                                             .iteration = iteration_config(),
                                         },
                                         rng),
               std::invalid_argument);
  EXPECT_THROW(
      (void)run_sr_optimization(lattice, condensate, parameterization, hamiltonian, state,
                                SrOptimizationConfig{
                                    .max_iterations = 1,
                                    .energy_tolerance = std::numeric_limits<double>::quiet_NaN(),
                                    .iteration = iteration_config(),
                                },
                                rng),
      std::invalid_argument);
}

}  // namespace
