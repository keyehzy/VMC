#include "vmc/sr_statistics.hpp"

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
using vmc::JastrowWaveFunction;
using vmc::Lattice;
using vmc::OccupancyConstraint;
using vmc::ProductWaveFunction;
using vmc::SrAccumulator;
using vmc::SrMeasurementResult;

std::vector<double> values_of(const Eigen::VectorXd& values) {
  return {values.data(), values.data() + values.size()};
}

MATCHER_P(DoubleNear, expected, "") {
  return testing::ExplainMatchResult(testing::DoubleNear(expected, 1.0e-12), arg, result_listener);
}

TEST(SrAccumulatorTest, EmptyAccumulatorHasZeroMeansCovarianceAndForces) {
  const SrAccumulator accumulator{2};

  EXPECT_EQ(accumulator.parameter_count(), 2);
  EXPECT_EQ(accumulator.sample_count(), 0);
  EXPECT_DOUBLE_EQ(accumulator.energy_mean(), 0.0);
  EXPECT_TRUE(accumulator.derivative_means().isZero());
  EXPECT_TRUE(accumulator.energy_derivative_means().isZero());
  EXPECT_TRUE(accumulator.derivative_outer_means().isZero());
  EXPECT_TRUE(accumulator.derivative_covariance().isZero());
  EXPECT_TRUE(accumulator.forces().isZero());
}

TEST(SrAccumulatorTest, ComputesMeansCovarianceAndForces) {
  SrAccumulator accumulator{2};
  Eigen::VectorXd first(2);
  first << 2.0, -1.0;
  Eigen::VectorXd second(2);
  second << 4.0, 1.0;
  Eigen::VectorXd third(2);
  third << 6.0, 3.0;

  accumulator.add(1.0, first);
  accumulator.add(3.0, second);
  accumulator.add(5.0, third);

  EXPECT_EQ(accumulator.sample_count(), 3);
  EXPECT_DOUBLE_EQ(accumulator.energy_mean(), 3.0);
  EXPECT_THAT(values_of(accumulator.derivative_means()),
              ElementsAre(DoubleNear(4.0), DoubleNear(1.0)));
  EXPECT_THAT(values_of(accumulator.energy_derivative_means()),
              ElementsAre(DoubleNear(44.0 / 3.0), DoubleNear(17.0 / 3.0)));

  Eigen::MatrixXd expected_outer_means(2, 2);
  expected_outer_means << 56.0 / 3.0, 20.0 / 3.0, 20.0 / 3.0, 11.0 / 3.0;
  EXPECT_TRUE(accumulator.derivative_outer_means().isApprox(expected_outer_means));

  Eigen::MatrixXd expected_covariance(2, 2);
  expected_covariance << 8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0;
  EXPECT_TRUE(accumulator.derivative_covariance().isApprox(expected_covariance));

  EXPECT_THAT(values_of(accumulator.forces()),
              ElementsAre(DoubleNear(-16.0 / 3.0), DoubleNear(-16.0 / 3.0)));
}

TEST(SrAccumulatorTest, RejectsInvalidSamples) {
  EXPECT_THROW(SrAccumulator{0}, std::invalid_argument);

  SrAccumulator accumulator{2};
  EXPECT_THROW(accumulator.add(std::numeric_limits<double>::infinity(), Eigen::VectorXd::Zero(2)),
               std::invalid_argument);
  EXPECT_THROW(accumulator.add(1.0, Eigen::VectorXd::Zero(3)), std::invalid_argument);

  Eigen::VectorXd nonfinite(2);
  nonfinite << 0.0, std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(accumulator.add(1.0, nonfinite), std::invalid_argument);
}

TEST(SrMeasurementTest, ZeroSamplesOnlyRunsThermalization) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(2);
  const FullMatrixJastrowParameterization parameterization{2};
  const JastrowWaveFunction jastrow{parameterization.matrix()};
  const ProductWaveFunction wave_function{{condensate, jastrow}};
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 2.0,
      .interaction_u = 0.0,
  };
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  const SrMeasurementResult result = measure_sr_statistics(lattice, wave_function, parameterization,
                                                           hamiltonian, state, 3, 0, 0, rng);

  EXPECT_EQ(result.thermalization_stats.attempted_steps, 3);
  EXPECT_EQ(result.thermalization_stats.proposed_steps, 3);
  EXPECT_EQ(result.thermalization_stats.accepted_steps, 3);
  EXPECT_EQ(result.sampling_stats.attempted_steps, 0);
  EXPECT_EQ(result.statistics.parameter_count(), 3);
  EXPECT_EQ(result.statistics.sample_count(), 0);
}

TEST(SrMeasurementTest, MeasuresEnergyAndLogDerivatives) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(2);
  const FullMatrixJastrowParameterization parameterization{2};
  const JastrowWaveFunction jastrow{parameterization.matrix()};
  const ProductWaveFunction wave_function{{condensate, jastrow}};
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 2.0,
      .interaction_u = 0.0,
  };
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  const SrMeasurementResult result = measure_sr_statistics(lattice, wave_function, parameterization,
                                                           hamiltonian, state, 2, 5, 1, rng);

  EXPECT_EQ(result.thermalization_stats.attempted_steps, 2);
  EXPECT_EQ(result.sampling_stats.attempted_steps, 5);
  EXPECT_EQ(result.sampling_stats.proposed_steps, 5);
  EXPECT_EQ(result.sampling_stats.accepted_steps, 5);
  EXPECT_EQ(result.statistics.sample_count(), 5);
  EXPECT_DOUBLE_EQ(result.statistics.energy_mean(), -2.0);
  EXPECT_THAT(values_of(result.statistics.derivative_means()),
              ElementsAre(DoubleNear(-0.2), DoubleNear(0.0), DoubleNear(-0.3)));
  EXPECT_TRUE(result.statistics.derivative_covariance().allFinite());
  EXPECT_TRUE(result.statistics.forces().isZero(1.0e-12));
}

TEST(SrMeasurementTest, RejectsInvalidInputs) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(2);
  const FullMatrixJastrowParameterization parameterization{2};
  const JastrowWaveFunction jastrow{parameterization.matrix()};
  const ProductWaveFunction wave_function{{condensate, jastrow}};
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 2.0,
      .interaction_u = 0.0,
  };
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  EXPECT_THROW(measure_sr_statistics(lattice, wave_function, parameterization, hamiltonian, state,
                                     0, 1, 0, rng),
               std::invalid_argument);

  const FullMatrixJastrowParameterization wrong_parameterization{3};
  EXPECT_THROW(measure_sr_statistics(lattice, wave_function, wrong_parameterization, hamiltonian,
                                     state, 0, 0, 0, rng),
               std::invalid_argument);
}

}  // namespace
