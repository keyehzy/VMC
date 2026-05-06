#include "vmc/hamiltonian.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace {

using vmc::BoseHubbardHamiltonian;
using vmc::BosonState;
using vmc::BoundaryCondition;
using vmc::CondensateWaveFunction;
using vmc::JastrowWaveFunction;
using vmc::Lattice;
using vmc::OccupancyConstraint;

TEST(HamiltonianTest, ZeroParametersGiveZeroLocalEnergy) {
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  const BosonState state = BosonState::from_occupations({2, 1}, OccupancyConstraint::SoftCore);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 0.0,
      .interaction_u = 0.0,
  };

  EXPECT_DOUBLE_EQ(local_energy(lattice, state, wave_function, hamiltonian), 0.0);
}

TEST(HamiltonianTest, ComputesSoftCorePotentialEnergy) {
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  const BosonState state = BosonState::from_occupations({2, 1}, OccupancyConstraint::SoftCore);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 0.0,
      .interaction_u = 3.0,
  };

  EXPECT_DOUBLE_EQ(local_energy(lattice, state, wave_function, hamiltonian), 3.0);
}

TEST(HamiltonianTest, HardCorePotentialEnergyIsZero) {
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  const BosonState state =
      BosonState::from_boson_positions(2, {0, 1}, OccupancyConstraint::HardCore);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 0.0,
      .interaction_u = 3.0,
  };

  EXPECT_DOUBLE_EQ(local_energy(lattice, state, wave_function, hamiltonian), 0.0);
}

TEST(HamiltonianTest, ComputesSingleBosonTwoSiteKineticEnergy) {
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  const BosonState state = BosonState::from_occupations({1, 0}, OccupancyConstraint::SoftCore);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 2.0,
      .interaction_u = 0.0,
  };

  EXPECT_DOUBLE_EQ(local_energy(lattice, state, wave_function, hamiltonian), -2.0);
}

TEST(HamiltonianTest, ComputesTwoBosonTwoSiteSoftCoreKineticEnergy) {
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  const BosonState state = BosonState::from_occupations({1, 1}, OccupancyConstraint::SoftCore);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 2.0,
      .interaction_u = 0.0,
  };

  EXPECT_DOUBLE_EQ(local_energy(lattice, state, wave_function, hamiltonian), -4.0);
}

TEST(HamiltonianTest, FullHardCoreTwoSiteChainHasNoKineticEnergy) {
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  const BosonState state =
      BosonState::from_boson_positions(2, {0, 1}, OccupancyConstraint::HardCore);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 2.0,
      .interaction_u = 0.0,
  };

  EXPECT_DOUBLE_EQ(local_energy(lattice, state, wave_function, hamiltonian), 0.0);
}

TEST(HamiltonianTest, IncludesWaveFunctionRatioInKineticEnergy) {
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(2);
  Eigen::MatrixXd parameters(2, 2);
  parameters << 0.0, 0.0, 0.0, 1.0;
  const JastrowWaveFunction jastrow{parameters};
  const vmc::ProductWaveFunction wave_function{{condensate, jastrow}};
  const BosonState state = BosonState::from_occupations({1, 0}, OccupancyConstraint::SoftCore);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 2.0,
      .interaction_u = 0.0,
  };

  EXPECT_DOUBLE_EQ(local_energy(lattice, state, wave_function, hamiltonian), -2.0 * std::exp(-0.5));
}

TEST(HamiltonianTest, RejectsMismatchedSizes) {
  const Lattice lattice = Lattice::chain(3, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  const BosonState state = BosonState::from_occupations({1, 0}, OccupancyConstraint::SoftCore);
  const BoseHubbardHamiltonian hamiltonian{
      .hopping_t = 1.0,
      .interaction_u = 1.0,
  };

  EXPECT_THROW((void)local_energy(lattice, state, wave_function, hamiltonian),
               std::invalid_argument);
}

TEST(HamiltonianTest, RejectsNonfiniteParameters) {
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  const BosonState state = BosonState::from_occupations({1, 0}, OccupancyConstraint::SoftCore);

  EXPECT_THROW((void)local_energy(lattice, state, wave_function,
                                  BoseHubbardHamiltonian{
                                      .hopping_t = std::numeric_limits<double>::infinity(),
                                      .interaction_u = 1.0,
                                  }),
               std::invalid_argument);
  EXPECT_THROW((void)local_energy(lattice, state, wave_function,
                                  BoseHubbardHamiltonian{
                                      .hopping_t = 1.0,
                                      .interaction_u = std::numeric_limits<double>::quiet_NaN(),
                                  }),
               std::invalid_argument);
}

}  // namespace
