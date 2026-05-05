#include "vmc/lattice.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace {

using testing::ElementsAre;
using vmc::BoundaryCondition;
using vmc::Lattice;

std::vector<Lattice::Site> neighbors_of(const Lattice& lattice, Lattice::Site site) {
  const auto neighbors = lattice.neighbors(site);
  return {neighbors.begin(), neighbors.end()};
}

std::vector<Lattice::Bond> bonds_of(const Lattice& lattice) {
  const auto bonds = lattice.bonds();
  return {bonds.begin(), bonds.end()};
}

TEST(LatticeTest, BuildsOpenChain) {
  const Lattice lattice = Lattice::chain(4, BoundaryCondition::Open);

  EXPECT_EQ(lattice.site_count(), 4);
  EXPECT_EQ(lattice.bond_count(), 3);
  EXPECT_THAT(neighbors_of(lattice, 0), ElementsAre(1));
  EXPECT_THAT(neighbors_of(lattice, 1), ElementsAre(0, 2));
  EXPECT_THAT(neighbors_of(lattice, 2), ElementsAre(1, 3));
  EXPECT_THAT(neighbors_of(lattice, 3), ElementsAre(2));
  EXPECT_THAT(bonds_of(lattice),
              ElementsAre(Lattice::Bond{0, 1}, Lattice::Bond{1, 2}, Lattice::Bond{2, 3}));
}

TEST(LatticeTest, BuildsPeriodicChain) {
  const Lattice lattice = Lattice::chain(4, BoundaryCondition::Periodic);

  EXPECT_EQ(lattice.site_count(), 4);
  EXPECT_EQ(lattice.bond_count(), 4);
  EXPECT_THAT(neighbors_of(lattice, 0), ElementsAre(1, 3));
  EXPECT_THAT(neighbors_of(lattice, 1), ElementsAre(0, 2));
  EXPECT_THAT(neighbors_of(lattice, 2), ElementsAre(1, 3));
  EXPECT_THAT(neighbors_of(lattice, 3), ElementsAre(0, 2));
  EXPECT_THAT(bonds_of(lattice), ElementsAre(Lattice::Bond{0, 1}, Lattice::Bond{0, 3},
                                             Lattice::Bond{1, 2}, Lattice::Bond{2, 3}));
}

TEST(LatticeTest, BuildsOpenSquareWithRowMajorSites) {
  const Lattice lattice = Lattice::square(2, 2, BoundaryCondition::Open);

  EXPECT_EQ(lattice.site_count(), 4);
  EXPECT_EQ(lattice.bond_count(), 4);
  EXPECT_THAT(neighbors_of(lattice, 0), ElementsAre(1, 2));
  EXPECT_THAT(neighbors_of(lattice, 1), ElementsAre(0, 3));
  EXPECT_THAT(neighbors_of(lattice, 2), ElementsAre(0, 3));
  EXPECT_THAT(neighbors_of(lattice, 3), ElementsAre(1, 2));
  EXPECT_THAT(bonds_of(lattice), ElementsAre(Lattice::Bond{0, 1}, Lattice::Bond{0, 2},
                                             Lattice::Bond{1, 3}, Lattice::Bond{2, 3}));
}

TEST(LatticeTest, BuildsPeriodicSquare) {
  const Lattice lattice = Lattice::square(3, 3, BoundaryCondition::Periodic);

  EXPECT_EQ(lattice.site_count(), 9);
  EXPECT_EQ(lattice.bond_count(), 18);

  for (Lattice::Site site = 0; site < lattice.site_count(); ++site) {
    EXPECT_EQ(lattice.neighbors(site).size(), 4);
  }

  EXPECT_THAT(neighbors_of(lattice, 0), ElementsAre(1, 2, 3, 6));
}

TEST(LatticeTest, ChecksNeighborRelation) {
  const Lattice lattice = Lattice::square(2, 2, BoundaryCondition::Open);

  EXPECT_TRUE(lattice.are_neighbors(0, 1));
  EXPECT_TRUE(lattice.are_neighbors(0, 2));
  EXPECT_FALSE(lattice.are_neighbors(0, 3));
}

TEST(LatticeTest, RejectsInvalidExtents) {
  EXPECT_THROW(Lattice::chain(0, BoundaryCondition::Open), std::invalid_argument);
  EXPECT_THROW(Lattice::chain(2, BoundaryCondition::Periodic), std::invalid_argument);
  EXPECT_THROW(Lattice::square(0, 2, BoundaryCondition::Open), std::invalid_argument);
  EXPECT_THROW(Lattice::square(2, 0, BoundaryCondition::Open), std::invalid_argument);
  EXPECT_THROW(Lattice::square(2, 3, BoundaryCondition::Periodic), std::invalid_argument);
  EXPECT_THROW(Lattice::square(3, 2, BoundaryCondition::Periodic), std::invalid_argument);
}

TEST(LatticeTest, RejectsOutOfRangeSiteAccess) {
  const Lattice lattice = Lattice::chain(3, BoundaryCondition::Open);

  EXPECT_THROW((void)lattice.neighbors(3), std::out_of_range);
  EXPECT_THROW((void)lattice.are_neighbors(0, 3), std::out_of_range);
}

}  // namespace
