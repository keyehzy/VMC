#include "vmc/proposal.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <vector>

namespace {

using testing::ElementsAre;
using vmc::BosonHop;
using vmc::BosonState;
using vmc::BoundaryCondition;
using vmc::Lattice;
using vmc::OccupancyConstraint;

std::vector<BosonHop> proposal_sequence(const Lattice& lattice, const BosonState& state,
                                        std::mt19937_64& rng, std::size_t count) {
  std::vector<BosonHop> proposals;
  proposals.reserve(count);

  for (std::size_t i = 0; i < count; ++i) {
    proposals.push_back(propose_nearest_neighbor_hop(lattice, state, rng).value());
  }

  return proposals;
}

TEST(ProposalTest, ReturnsNoProposalWhenThereAreNoBosons) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(4, BoundaryCondition::Open);
  const BosonState state = BosonState::empty(4, OccupancyConstraint::SoftCore);

  EXPECT_EQ(propose_nearest_neighbor_hop(lattice, state, rng), std::nullopt);
}

TEST(ProposalTest, ReturnsNoProposalWhenSelectedBosonHasNoNeighbors) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(1, BoundaryCondition::Open);
  const BosonState state = BosonState::from_boson_positions(1, {0}, OccupancyConstraint::SoftCore);

  EXPECT_EQ(propose_nearest_neighbor_hop(lattice, state, rng), std::nullopt);
}

TEST(ProposalTest, ProposesNearestNeighborHop) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(4, BoundaryCondition::Open);
  const BosonState state =
      BosonState::from_boson_positions(4, {0, 2}, OccupancyConstraint::SoftCore);

  const std::optional<BosonHop> proposal = propose_nearest_neighbor_hop(lattice, state, rng);

  ASSERT_TRUE(proposal.has_value());
  EXPECT_LT(proposal->boson, state.boson_count());
  EXPECT_EQ(proposal->source, state.boson_position(proposal->boson));
  EXPECT_TRUE(lattice.are_neighbors(proposal->source, proposal->destination));
}

TEST(ProposalTest, AllowsHardCoreProposalIntoOccupiedDestination) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(2, BoundaryCondition::Open);
  const BosonState state =
      BosonState::from_boson_positions(2, {0, 1}, OccupancyConstraint::HardCore);

  const std::optional<BosonHop> proposal = propose_nearest_neighbor_hop(lattice, state, rng);

  ASSERT_TRUE(proposal.has_value());
  EXPECT_TRUE(lattice.are_neighbors(proposal->source, proposal->destination));
  EXPECT_GT(state.occupation(proposal->destination), 0);
  EXPECT_FALSE(state.can_move(proposal->boson, proposal->destination));
}

TEST(ProposalTest, RejectsMismatchedLatticeAndStateSizes) {
  std::mt19937_64 rng{1234};
  const Lattice lattice = Lattice::chain(4, BoundaryCondition::Open);
  const BosonState state = BosonState::from_boson_positions(3, {0}, OccupancyConstraint::SoftCore);

  EXPECT_THROW((void)propose_nearest_neighbor_hop(lattice, state, rng), std::invalid_argument);
}

TEST(ProposalTest, SameSeedProducesSameProposalSequence) {
  std::mt19937_64 first_rng{1234};
  std::mt19937_64 second_rng{1234};
  const Lattice lattice = Lattice::square(3, 3, BoundaryCondition::Periodic);
  const BosonState state =
      BosonState::from_boson_positions(9, {0, 2, 4, 6}, OccupancyConstraint::HardCore);

  EXPECT_THAT(proposal_sequence(lattice, state, first_rng, 8),
              testing::ContainerEq(proposal_sequence(lattice, state, second_rng, 8)));
}

TEST(ProposalTest, DifferentSeedsCanProduceDifferentProposalSequences) {
  std::mt19937_64 first_rng{1234};
  std::mt19937_64 second_rng{5678};
  const Lattice lattice = Lattice::square(3, 3, BoundaryCondition::Periodic);
  const BosonState state =
      BosonState::from_boson_positions(9, {0, 2, 4, 6}, OccupancyConstraint::HardCore);

  EXPECT_NE(proposal_sequence(lattice, state, first_rng, 8),
            proposal_sequence(lattice, state, second_rng, 8));
}

TEST(ProposalTest, BosonHopHasValueEquality) {
  EXPECT_EQ((BosonHop{.boson = 1, .source = 2, .destination = 3}),
            (BosonHop{.boson = 1, .source = 2, .destination = 3}));
  EXPECT_NE((BosonHop{.boson = 1, .source = 2, .destination = 3}),
            (BosonHop{.boson = 1, .source = 3, .destination = 2}));
}

}  // namespace
