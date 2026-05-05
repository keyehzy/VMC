#include "vmc/initialization.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using testing::Each;
using testing::ElementsAre;
using vmc::BosonState;
using vmc::BoundaryCondition;
using vmc::InitialBosonConfig;
using vmc::Lattice;
using vmc::OccupancyConstraint;

std::vector<std::size_t> occupations_of(const BosonState& state) {
  const auto occupations = state.occupations();
  return {occupations.begin(), occupations.end()};
}

std::vector<BosonState::Site> positions_of(const BosonState& state) {
  const auto positions = state.boson_positions();
  return {positions.begin(), positions.end()};
}

std::size_t total_occupation(const BosonState& state) {
  const auto occupations = state.occupations();
  return std::accumulate(occupations.begin(), occupations.end(), std::size_t{0});
}

TEST(InitializationTest, InitializesSoftCoreStateWithRequestedCounts) {
  std::mt19937_64 rng{1234};

  const BosonState state = initialize_boson_state(
      InitialBosonConfig{
          .site_count = 5, .boson_count = 20, .constraint = OccupancyConstraint::SoftCore},
      rng);

  EXPECT_EQ(state.site_count(), 5);
  EXPECT_EQ(state.boson_count(), 20);
  EXPECT_EQ(state.constraint(), OccupancyConstraint::SoftCore);
  EXPECT_EQ(total_occupation(state), 20);

  for (const BosonState::Site site : state.boson_positions()) {
    EXPECT_LT(site, state.site_count());
  }
}

TEST(InitializationTest, InitializesHardCoreStateWithAtMostOneBosonPerSite) {
  std::mt19937_64 rng{1234};

  const BosonState state = initialize_boson_state(
      InitialBosonConfig{
          .site_count = 8, .boson_count = 5, .constraint = OccupancyConstraint::HardCore},
      rng);

  EXPECT_EQ(state.site_count(), 8);
  EXPECT_EQ(state.boson_count(), 5);
  EXPECT_EQ(state.constraint(), OccupancyConstraint::HardCore);
  EXPECT_EQ(total_occupation(state), 5);
  EXPECT_THAT(occupations_of(state), Each(testing::AnyOf(0, 1)));
}

TEST(InitializationTest, AllowsZeroBosons) {
  std::mt19937_64 rng{1234};

  const BosonState softcore = initialize_boson_state(
      InitialBosonConfig{
          .site_count = 4, .boson_count = 0, .constraint = OccupancyConstraint::SoftCore},
      rng);
  const BosonState hardcore = initialize_boson_state(
      InitialBosonConfig{
          .site_count = 4, .boson_count = 0, .constraint = OccupancyConstraint::HardCore},
      rng);

  EXPECT_THAT(occupations_of(softcore), ElementsAre(0, 0, 0, 0));
  EXPECT_THAT(positions_of(softcore), testing::IsEmpty());
  EXPECT_THAT(occupations_of(hardcore), ElementsAre(0, 0, 0, 0));
  EXPECT_THAT(positions_of(hardcore), testing::IsEmpty());
}

TEST(InitializationTest, RejectsInvalidConfigurations) {
  std::mt19937_64 rng{1234};

  EXPECT_THROW(
      initialize_boson_state(
          InitialBosonConfig{
              .site_count = 0, .boson_count = 0, .constraint = OccupancyConstraint::SoftCore},
          rng),
      std::invalid_argument);
  EXPECT_THROW(
      initialize_boson_state(
          InitialBosonConfig{
              .site_count = 2, .boson_count = 3, .constraint = OccupancyConstraint::HardCore},
          rng),
      std::invalid_argument);
}

TEST(InitializationTest, SameSeedProducesSameState) {
  std::mt19937_64 first_rng{1234};
  std::mt19937_64 second_rng{1234};
  const InitialBosonConfig config{
      .site_count = 10,
      .boson_count = 6,
      .constraint = OccupancyConstraint::HardCore,
  };

  const BosonState first = initialize_boson_state(config, first_rng);
  const BosonState second = initialize_boson_state(config, second_rng);

  EXPECT_THAT(positions_of(first), testing::ContainerEq(positions_of(second)));
  EXPECT_THAT(occupations_of(first), testing::ContainerEq(occupations_of(second)));
}

TEST(InitializationTest, DifferentSeedsCanProduceDifferentStates) {
  std::mt19937_64 first_rng{1234};
  std::mt19937_64 second_rng{5678};
  const InitialBosonConfig config{
      .site_count = 20,
      .boson_count = 10,
      .constraint = OccupancyConstraint::HardCore,
  };

  const BosonState first = initialize_boson_state(config, first_rng);
  const BosonState second = initialize_boson_state(config, second_rng);

  EXPECT_NE(positions_of(first), positions_of(second));
}

TEST(InitializationTest, UsesLatticeSiteCount) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(6, BoundaryCondition::Open);

  const BosonState state = initialize_boson_state(lattice, 4, OccupancyConstraint::HardCore, rng);

  EXPECT_EQ(state.site_count(), lattice.site_count());
  EXPECT_EQ(state.boson_count(), 4);
  EXPECT_THAT(occupations_of(state), Each(testing::AnyOf(0, 1)));
}

}  // namespace
