#include "vmc/sr_iteration.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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
using vmc::SrIterationResult;
using vmc::SrUpdateConfig;

std::vector<double> values_of(const Eigen::VectorXd& values) {
  return {values.data(), values.data() + values.size()};
}

MATCHER_P(DoubleNear, expected, "") {
  return testing::ExplainMatchResult(testing::DoubleNear(expected, 1.0e-12), arg, result_listener);
}

TEST(SrIterationTest, RunsOneIterationAndAppliesUpdate) {
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

  const SrIterationResult result =
      run_sr_iteration(lattice, condensate, parameterization, hamiltonian, state,
                       SrIterationConfig{
                           .thermalization_sweeps = 2,
                           .sample_count = 5,
                           .sweeps_between_samples = 1,
                           .update =
                               SrUpdateConfig{
                                   .step_size = 0.25,
                                   .diagonal_shift = 1.0,
                               },
                       },
                       rng);

  EXPECT_EQ(result.measurement.thermalization_stats.attempted_steps, 2);
  EXPECT_EQ(result.measurement.sampling_stats.attempted_steps, 5);
  EXPECT_EQ(result.measurement.statistics.sample_count(), 5);
  EXPECT_DOUBLE_EQ(result.measurement.statistics.energy_mean(), 0.0);
  EXPECT_TRUE(result.update.forces.isZero());
  EXPECT_TRUE(result.update.delta.isZero());
  EXPECT_THAT(values_of(result.parameters_before),
              ElementsAre(DoubleNear(0.25), DoubleNear(-0.5), DoubleNear(1.0)));
  EXPECT_THAT(values_of(result.parameters_after),
              ElementsAre(DoubleNear(0.25), DoubleNear(-0.5), DoubleNear(1.0)));
  EXPECT_THAT(values_of(parameterization.parameters()),
              ElementsAre(DoubleNear(0.25), DoubleNear(-0.5), DoubleNear(1.0)));
}

TEST(SrIterationTest, RejectsInvalidMeasurementConfig) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(2);
  FullMatrixJastrowParameterization parameterization{2};
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 0.0,
      .interaction_u = 0.0,
  };
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  EXPECT_THROW((void)run_sr_iteration(lattice, condensate, parameterization, hamiltonian, state,
                                      SrIterationConfig{
                                          .thermalization_sweeps = 0,
                                          .sample_count = 1,
                                          .sweeps_between_samples = 0,
                                          .update =
                                              SrUpdateConfig{
                                                  .step_size = 0.25,
                                                  .diagonal_shift = 1.0,
                                              },
                                      },
                                      rng),
               std::invalid_argument);
}

TEST(SrIterationTest, RejectsInvalidUpdateConfigWithoutChangingParameters) {
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

  EXPECT_THROW((void)run_sr_iteration(lattice, condensate, parameterization, hamiltonian, state,
                                      SrIterationConfig{
                                          .thermalization_sweeps = 0,
                                          .sample_count = 1,
                                          .sweeps_between_samples = 1,
                                          .update =
                                              SrUpdateConfig{
                                                  .step_size = -0.25,
                                                  .diagonal_shift = 1.0,
                                              },
                                      },
                                      rng),
               std::invalid_argument);
  EXPECT_THAT(values_of(parameterization.parameters()),
              ElementsAre(DoubleNear(0.25), DoubleNear(-0.5), DoubleNear(1.0)));
}

}  // namespace
