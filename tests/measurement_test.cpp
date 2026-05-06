#include "vmc/measurement.hpp"

#include <gtest/gtest.h>

#include <random>
#include <stdexcept>

namespace {

using vmc::BoseHubbardHamiltonian;
using vmc::BosonState;
using vmc::BoundaryCondition;
using vmc::CondensateWaveFunction;
using vmc::EnergyRunResult;
using vmc::Lattice;
using vmc::OccupancyConstraint;

TEST(MeasurementTest, ZeroSamplesOnlyRunsThermalization) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 2.0,
      .interaction_u = 0.0,
  };
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  const EnergyRunResult result =
      measure_local_energy(lattice, wave_function, hamiltonian, state, 3, 0, 0, rng);

  EXPECT_EQ(result.thermalization_stats.attempted_steps, 3);
  EXPECT_EQ(result.thermalization_stats.proposed_steps, 3);
  EXPECT_EQ(result.thermalization_stats.accepted_steps, 3);
  EXPECT_EQ(result.sampling_stats.attempted_steps, 0);
  EXPECT_EQ(result.sampling_stats.proposed_steps, 0);
  EXPECT_EQ(result.sampling_stats.accepted_steps, 0);
  EXPECT_EQ(result.energy.count(), 0);
}

TEST(MeasurementTest, RejectsSamplesWithoutSweepsBetweenSamples) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 2.0,
      .interaction_u = 0.0,
  };
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  EXPECT_THROW(measure_local_energy(lattice, wave_function, hamiltonian, state, 0, 1, 0, rng),
               std::invalid_argument);
}

TEST(MeasurementTest, MeasuresConstantTwoSiteOneBosonEnergy) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 2.0,
      .interaction_u = 0.0,
  };
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  const EnergyRunResult result =
      measure_local_energy(lattice, wave_function, hamiltonian, state, 2, 5, 3, rng);

  EXPECT_EQ(result.thermalization_stats.attempted_steps, 2);
  EXPECT_EQ(result.sampling_stats.attempted_steps, 15);
  EXPECT_EQ(result.sampling_stats.proposed_steps, 15);
  EXPECT_EQ(result.sampling_stats.accepted_steps, 15);
  EXPECT_EQ(result.energy.count(), 5);
  EXPECT_DOUBLE_EQ(result.energy.mean(), -2.0);
  EXPECT_DOUBLE_EQ(result.energy.variance(), 0.0);
  EXPECT_DOUBLE_EQ(result.energy.standard_error(), 0.0);
}

TEST(MeasurementTest, ZeroBosonSamplingRecordsStaticEnergy) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(3, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(3);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 2.0,
      .interaction_u = 3.0,
  };
  BosonState state = BosonState::empty(3, OccupancyConstraint::SoftCore);

  const EnergyRunResult result =
      measure_local_energy(lattice, wave_function, hamiltonian, state, 4, 3, 2, rng);

  EXPECT_EQ(result.thermalization_stats.attempted_steps, 0);
  EXPECT_EQ(result.sampling_stats.attempted_steps, 0);
  EXPECT_EQ(result.energy.count(), 3);
  EXPECT_DOUBLE_EQ(result.energy.mean(), 0.0);
}

}  // namespace
