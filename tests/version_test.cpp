#include "vmc/version.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace {

TEST(VersionTest, ReportsProjectVersion) {
  EXPECT_THAT(vmc::version(), testing::StrEq("0.1.0"));
}

TEST(EigenTest, IsAvailableToProjectTargets) {
  const Eigen::Vector2d value{3.0, 4.0};
  EXPECT_DOUBLE_EQ(value.norm(), 5.0);
}

}  // namespace
