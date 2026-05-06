#include "vmc/boson_state.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace {

using testing::ElementsAre;
using testing::IsEmpty;
using vmc::BosonState;
using vmc::OccupancyConstraint;

std::vector<std::size_t> occupations_of(const BosonState& state) {
  const auto occupations = state.occupations();
  return {occupations.begin(), occupations.end()};
}

std::vector<BosonState::Site> positions_of(const BosonState& state) {
  const auto positions = state.boson_positions();
  return {positions.begin(), positions.end()};
}

TEST(BosonStateTest, BuildsEmptySoftCoreState) {
  const BosonState state = BosonState::empty(4, OccupancyConstraint::SoftCore);

  EXPECT_EQ(state.site_count(), 4);
  EXPECT_EQ(state.boson_count(), 0);
  EXPECT_EQ(state.constraint(), OccupancyConstraint::SoftCore);
  EXPECT_FALSE(state.is_hardcore());
  EXPECT_THAT(occupations_of(state), ElementsAre(0, 0, 0, 0));
  EXPECT_THAT(positions_of(state), IsEmpty());
}

TEST(BosonStateTest, BuildsEmptyHardCoreState) {
  const BosonState state = BosonState::empty(3, OccupancyConstraint::HardCore);

  EXPECT_EQ(state.site_count(), 3);
  EXPECT_EQ(state.boson_count(), 0);
  EXPECT_EQ(state.constraint(), OccupancyConstraint::HardCore);
  EXPECT_TRUE(state.is_hardcore());
  EXPECT_THAT(occupations_of(state), ElementsAre(0, 0, 0));
  EXPECT_THAT(positions_of(state), IsEmpty());
}

TEST(BosonStateTest, BuildsFromOccupations) {
  BosonState state = BosonState::from_occupations({2, 0, 1, 3}, OccupancyConstraint::SoftCore);

  EXPECT_EQ(state.site_count(), 4);
  EXPECT_EQ(state.boson_count(), 6);
  EXPECT_THAT(occupations_of(state), ElementsAre(2, 0, 1, 3));
  EXPECT_THAT(positions_of(state), ElementsAre(0, 0, 2, 3, 3, 3));
}

TEST(BosonStateTest, BuildsFromBosonPositions) {
  BosonState state =
      BosonState::from_boson_positions(4, {3, 0, 3, 2}, OccupancyConstraint::SoftCore);

  EXPECT_EQ(state.site_count(), 4);
  EXPECT_EQ(state.boson_count(), 4);
  EXPECT_THAT(occupations_of(state), ElementsAre(1, 0, 1, 2));
  EXPECT_THAT(positions_of(state), ElementsAre(3, 0, 3, 2));
}

TEST(BosonStateTest, EnforcesHardCoreConstruction) {
  EXPECT_THROW(BosonState::from_occupations({1, 2}, OccupancyConstraint::HardCore),
               std::invalid_argument);
  EXPECT_THROW(BosonState::from_boson_positions(3, {0, 1, 1}, OccupancyConstraint::HardCore),
               std::invalid_argument);
}

TEST(BosonStateTest, AllowsSoftCoreDoubleOccupation) {
  const BosonState state =
      BosonState::from_boson_positions(3, {1, 1}, OccupancyConstraint::SoftCore);

  EXPECT_THAT(occupations_of(state), ElementsAre(0, 2, 0));
  EXPECT_THAT(positions_of(state), ElementsAre(1, 1));
}

TEST(BosonStateTest, ChecksMoveValidity) {
  const BosonState state =
      BosonState::from_boson_positions(3, {0, 1}, OccupancyConstraint::HardCore);

  EXPECT_FALSE(state.can_move(0, 0));
  EXPECT_FALSE(state.can_move(0, 1));
  EXPECT_TRUE(state.can_move(0, 2));
}

TEST(BosonStateTest, MovesBosonAndUpdatesBothRepresentations) {
  BosonState state = BosonState::from_boson_positions(4, {0, 2}, OccupancyConstraint::HardCore);

  state.move_boson(1, 3);

  EXPECT_THAT(occupations_of(state), ElementsAre(1, 0, 0, 1));
  EXPECT_THAT(positions_of(state), ElementsAre(0, 3));
  EXPECT_EQ(state.occupation(2), 0);
  EXPECT_EQ(state.occupation(3), 1);
  EXPECT_EQ(state.boson_position(1), 3);
}

TEST(BosonStateTest, MovesSoftCoreBosonIntoOccupiedSite) {
  BosonState state = BosonState::from_boson_positions(3, {0, 1}, OccupancyConstraint::SoftCore);

  state.move_boson(0, 1);

  EXPECT_THAT(occupations_of(state), ElementsAre(0, 2, 0));
  EXPECT_THAT(positions_of(state), ElementsAre(1, 1));
}

TEST(BosonStateTest, FindsFirstBosonAtSite) {
  const BosonState state =
      BosonState::from_boson_positions(4, {3, 1, 3, 2}, OccupancyConstraint::SoftCore);

  EXPECT_EQ(state.first_boson_at(1), 1);
  EXPECT_EQ(state.first_boson_at(2), 3);
  EXPECT_EQ(state.first_boson_at(3), 0);
  EXPECT_EQ(state.first_boson_at(0), std::nullopt);
}

TEST(BosonStateTest, RejectsInvalidConstructionInputs) {
  EXPECT_THROW(BosonState::empty(0, OccupancyConstraint::SoftCore), std::invalid_argument);
  EXPECT_THROW(BosonState::from_occupations({}, OccupancyConstraint::SoftCore),
               std::invalid_argument);
  EXPECT_THROW(BosonState::from_boson_positions(0, {}, OccupancyConstraint::SoftCore),
               std::invalid_argument);
  EXPECT_THROW(BosonState::from_boson_positions(2, {0, 2}, OccupancyConstraint::SoftCore),
               std::out_of_range);
}

TEST(BosonStateTest, RejectsInvalidSiteAndBosonAccess) {
  BosonState state = BosonState::from_boson_positions(3, {0}, OccupancyConstraint::HardCore);

  EXPECT_THROW((void)state.occupation(3), std::out_of_range);
  EXPECT_THROW((void)state.boson_position(1), std::out_of_range);
  EXPECT_THROW((void)state.first_boson_at(3), std::out_of_range);
  EXPECT_THROW((void)state.can_move(1, 2), std::out_of_range);
  EXPECT_THROW((void)state.can_move(0, 3), std::out_of_range);
}

TEST(BosonStateTest, RejectsInvalidMoves) {
  BosonState state = BosonState::from_boson_positions(3, {0, 1}, OccupancyConstraint::HardCore);

  EXPECT_THROW(state.move_boson(0, 0), std::invalid_argument);
  EXPECT_THROW(state.move_boson(0, 1), std::invalid_argument);
  EXPECT_THROW(state.move_boson(2, 1), std::out_of_range);
  EXPECT_THROW(state.move_boson(0, 3), std::out_of_range);
}

}  // namespace
