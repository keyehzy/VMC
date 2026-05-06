#include "vmc/metropolis.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using testing::ElementsAre;
using vmc::BosonHop;
using vmc::BosonState;
using vmc::BoundaryCondition;
using vmc::CondensateWaveFunction;
using vmc::JastrowWaveFunction;
using vmc::Lattice;
using vmc::MetropolisRunStats;
using vmc::MetropolisStepResult;
using vmc::OccupancyConstraint;
using vmc::WaveFunction;

class FixedLogRatioWaveFunction final : public WaveFunction {
 public:
  FixedLogRatioWaveFunction(std::size_t site_count, double log_ratio)
      : site_count_{site_count}, log_ratio_{log_ratio} {}

  [[nodiscard]] std::size_t site_count() const override {
    return site_count_;
  }

  [[nodiscard]] double log_amplitude(const BosonState&) const override {
    return 0.0;
  }

  [[nodiscard]] double log_ratio(const BosonState&, const BosonHop&) const override {
    return log_ratio_;
  }

 private:
  std::size_t site_count_;
  double log_ratio_;
};

std::vector<std::size_t> occupations_of(const BosonState& state) {
  const auto occupations = state.occupations();
  return {occupations.begin(), occupations.end()};
}

std::vector<BosonState::Site> positions_of(const BosonState& state) {
  const auto positions = state.boson_positions();
  return {positions.begin(), positions.end()};
}

TEST(MetropolisTest, ReturnsNoProposalForZeroBosons) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(3, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(3);
  BosonState state = BosonState::empty(3, OccupancyConstraint::SoftCore);

  const MetropolisStepResult result = metropolis_step(lattice, wave_function, state, rng);

  EXPECT_FALSE(result.proposed);
  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.hop, std::nullopt);
  EXPECT_DOUBLE_EQ(result.acceptance_probability, 0.0);
  EXPECT_THAT(occupations_of(state), ElementsAre(0, 0, 0));
}

TEST(MetropolisTest, ReturnsNoProposalForIsolatedBoson) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(1, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(1);
  BosonState state = BosonState::from_boson_positions(1, {0}, OccupancyConstraint::SoftCore);

  const MetropolisStepResult result = metropolis_step(lattice, wave_function, state, rng);

  EXPECT_FALSE(result.proposed);
  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.hop, std::nullopt);
  EXPECT_DOUBLE_EQ(result.acceptance_probability, 0.0);
  EXPECT_THAT(occupations_of(state), ElementsAre(1));
  EXPECT_THAT(positions_of(state), ElementsAre(0));
}

TEST(MetropolisTest, ImmediatelyRejectsOccupiedHardCoreDestination) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  BosonState state = BosonState::from_boson_positions(2, {0, 1}, OccupancyConstraint::HardCore);

  const MetropolisStepResult result = metropolis_step(lattice, wave_function, state, rng);

  ASSERT_TRUE(result.proposed);
  ASSERT_TRUE(result.hop.has_value());
  EXPECT_FALSE(result.accepted);
  EXPECT_DOUBLE_EQ(result.acceptance_probability, 0.0);
  EXPECT_GT(state.occupation(result.hop->destination), 0);
  EXPECT_THAT(occupations_of(state), ElementsAre(1, 1));
  EXPECT_THAT(positions_of(state), ElementsAre(0, 1));
}

TEST(MetropolisTest, AcceptsGuaranteedMoveAndUpdatesState) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  const MetropolisStepResult result = metropolis_step(lattice, wave_function, state, rng);

  ASSERT_TRUE(result.proposed);
  ASSERT_TRUE(result.hop.has_value());
  EXPECT_TRUE(result.accepted);
  EXPECT_DOUBLE_EQ(result.acceptance_probability, 1.0);
  EXPECT_EQ(result.hop->source, 0);
  EXPECT_EQ(result.hop->destination, 1);
  EXPECT_THAT(occupations_of(state), ElementsAre(0, 1));
  EXPECT_THAT(positions_of(state), ElementsAre(1));
}

TEST(MetropolisTest, ReportsProbabilityBetweenZeroAndOne) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(2);
  Eigen::MatrixXd parameters(2, 2);
  parameters << 0.0, 0.0, 0.0, 1.0;
  const JastrowWaveFunction jastrow{parameters};
  const vmc::ProductWaveFunction wave_function{{condensate, jastrow}};
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  const MetropolisStepResult result = metropolis_step(lattice, wave_function, state, rng);

  ASSERT_TRUE(result.proposed);
  EXPECT_GT(result.acceptance_probability, 0.0);
  EXPECT_LE(result.acceptance_probability, 1.0);
}

TEST(MetropolisTest, RejectsMoveWhenAcceptanceDrawFails) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const FixedLogRatioWaveFunction wave_function{2, -20.0};
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  const MetropolisStepResult result = metropolis_step(lattice, wave_function, state, rng);

  ASSERT_TRUE(result.proposed);
  EXPECT_FALSE(result.accepted);
  EXPECT_DOUBLE_EQ(result.acceptance_probability, std::exp(-40.0));
  EXPECT_THAT(occupations_of(state), ElementsAre(1, 0));
  EXPECT_THAT(positions_of(state), ElementsAre(0));
}

TEST(MetropolisTest, SameSeedProducesSameTransition) {
  std::mt19937_64 first_rng{1234};
  std::mt19937_64 second_rng{1234};
  const Lattice lattice = Lattice::square(3, 3, BoundaryCondition::Periodic);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(9);
  BosonState first_state =
      BosonState::from_boson_positions(9, {0, 2, 4}, OccupancyConstraint::SoftCore);
  BosonState second_state =
      BosonState::from_boson_positions(9, {0, 2, 4}, OccupancyConstraint::SoftCore);

  const MetropolisStepResult first =
      metropolis_step(lattice, wave_function, first_state, first_rng);
  const MetropolisStepResult second =
      metropolis_step(lattice, wave_function, second_state, second_rng);

  EXPECT_EQ(first.proposed, second.proposed);
  EXPECT_EQ(first.accepted, second.accepted);
  EXPECT_EQ(first.hop, second.hop);
  EXPECT_DOUBLE_EQ(first.acceptance_probability, second.acceptance_probability);
  EXPECT_THAT(occupations_of(first_state), testing::ContainerEq(occupations_of(second_state)));
  EXPECT_THAT(positions_of(first_state), testing::ContainerEq(positions_of(second_state)));
}

TEST(MetropolisTest, EmptyRunStatsHaveZeroRates) {
  const MetropolisRunStats stats{
      .attempted_steps = 0,
      .proposed_steps = 0,
      .accepted_steps = 0,
  };

  EXPECT_DOUBLE_EQ(stats.proposal_rate(), 0.0);
  EXPECT_DOUBLE_EQ(stats.acceptance_rate(), 0.0);
}

TEST(MetropolisTest, RunStatsComputeRates) {
  const MetropolisRunStats stats{
      .attempted_steps = 8,
      .proposed_steps = 6,
      .accepted_steps = 3,
  };

  EXPECT_DOUBLE_EQ(stats.proposal_rate(), 0.75);
  EXPECT_DOUBLE_EQ(stats.acceptance_rate(), 0.5);
}

TEST(MetropolisTest, RunStatsAccumulate) {
  MetropolisRunStats stats{
      .attempted_steps = 8,
      .proposed_steps = 6,
      .accepted_steps = 3,
  };
  const MetropolisRunStats other{
      .attempted_steps = 2,
      .proposed_steps = 1,
      .accepted_steps = 1,
  };

  stats += other;

  EXPECT_EQ(stats.attempted_steps, 10);
  EXPECT_EQ(stats.proposed_steps, 7);
  EXPECT_EQ(stats.accepted_steps, 4);
  EXPECT_DOUBLE_EQ(stats.proposal_rate(), 0.7);
  EXPECT_DOUBLE_EQ(stats.acceptance_rate(), 4.0 / 7.0);
}

TEST(MetropolisTest, RunStatsAdd) {
  const MetropolisRunStats lhs{
      .attempted_steps = 8,
      .proposed_steps = 6,
      .accepted_steps = 3,
  };
  const MetropolisRunStats rhs{
      .attempted_steps = 2,
      .proposed_steps = 1,
      .accepted_steps = 1,
  };

  const MetropolisRunStats stats = lhs + rhs;

  EXPECT_EQ(stats.attempted_steps, 10);
  EXPECT_EQ(stats.proposed_steps, 7);
  EXPECT_EQ(stats.accepted_steps, 4);
}

TEST(MetropolisTest, RunStepsWithZeroStepsDoesNothing) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  const MetropolisRunStats stats = run_metropolis_steps(lattice, wave_function, state, 0, rng);

  EXPECT_EQ(stats.attempted_steps, 0);
  EXPECT_EQ(stats.proposed_steps, 0);
  EXPECT_EQ(stats.accepted_steps, 0);
  EXPECT_THAT(occupations_of(state), ElementsAre(1, 0));
  EXPECT_THAT(positions_of(state), ElementsAre(0));
}

TEST(MetropolisTest, RunStepsCountsIsolatedBosonAttemptsWithoutProposals) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(1, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(1);
  BosonState state = BosonState::from_boson_positions(1, {0}, OccupancyConstraint::SoftCore);

  const MetropolisRunStats stats = run_metropolis_steps(lattice, wave_function, state, 5, rng);

  EXPECT_EQ(stats.attempted_steps, 5);
  EXPECT_EQ(stats.proposed_steps, 0);
  EXPECT_EQ(stats.accepted_steps, 0);
  EXPECT_DOUBLE_EQ(stats.proposal_rate(), 0.0);
  EXPECT_DOUBLE_EQ(stats.acceptance_rate(), 0.0);
  EXPECT_THAT(occupations_of(state), ElementsAre(1));
  EXPECT_THAT(positions_of(state), ElementsAre(0));
}

TEST(MetropolisTest, RunStepsCountsGuaranteedAcceptedMoves) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);

  const MetropolisRunStats stats = run_metropolis_steps(lattice, wave_function, state, 4, rng);

  EXPECT_EQ(stats.attempted_steps, 4);
  EXPECT_EQ(stats.proposed_steps, 4);
  EXPECT_EQ(stats.accepted_steps, 4);
  EXPECT_DOUBLE_EQ(stats.proposal_rate(), 1.0);
  EXPECT_DOUBLE_EQ(stats.acceptance_rate(), 1.0);
}

TEST(MetropolisTest, RunStepsCountsHardCoreRejections) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  BosonState state = BosonState::from_boson_positions(2, {0, 1}, OccupancyConstraint::HardCore);

  const MetropolisRunStats stats = run_metropolis_steps(lattice, wave_function, state, 4, rng);

  EXPECT_EQ(stats.attempted_steps, 4);
  EXPECT_EQ(stats.proposed_steps, 4);
  EXPECT_EQ(stats.accepted_steps, 0);
  EXPECT_DOUBLE_EQ(stats.proposal_rate(), 1.0);
  EXPECT_DOUBLE_EQ(stats.acceptance_rate(), 0.0);
  EXPECT_THAT(occupations_of(state), ElementsAre(1, 1));
  EXPECT_THAT(positions_of(state), ElementsAre(0, 1));
}

TEST(MetropolisTest, RunSweepsAttemptsSweepCountTimesBosonCountSteps) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::square(3, 3, BoundaryCondition::Periodic);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(9);
  BosonState state = BosonState::from_boson_positions(9, {0, 2, 4}, OccupancyConstraint::SoftCore);

  const MetropolisRunStats stats = run_metropolis_sweeps(lattice, wave_function, state, 7, rng);

  EXPECT_EQ(stats.attempted_steps, 21);
  EXPECT_EQ(stats.proposed_steps, 21);
}

TEST(MetropolisTest, RunSweepsWithZeroBosonsAttemptsZeroSteps) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(3, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(3);
  BosonState state = BosonState::empty(3, OccupancyConstraint::SoftCore);

  const MetropolisRunStats stats = run_metropolis_sweeps(lattice, wave_function, state, 7, rng);

  EXPECT_EQ(stats.attempted_steps, 0);
  EXPECT_EQ(stats.proposed_steps, 0);
  EXPECT_EQ(stats.accepted_steps, 0);
}

TEST(MetropolisTest, RunSweepsRejectsStepCountOverflow) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(2);
  BosonState state = BosonState::from_boson_positions(2, {0, 1}, OccupancyConstraint::SoftCore);

  EXPECT_THROW(run_metropolis_sweeps(lattice, wave_function, state,
                                     std::numeric_limits<std::size_t>::max(), rng),
               std::overflow_error);
}

}  // namespace
