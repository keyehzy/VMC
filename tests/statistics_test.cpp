#include "vmc/statistics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

using vmc::RunningStats;

TEST(StatisticsTest, EmptyStatsAreZero) {
  const RunningStats stats;

  EXPECT_EQ(stats.count(), 0);
  EXPECT_DOUBLE_EQ(stats.mean(), 0.0);
  EXPECT_DOUBLE_EQ(stats.variance(), 0.0);
  EXPECT_DOUBLE_EQ(stats.standard_deviation(), 0.0);
  EXPECT_DOUBLE_EQ(stats.standard_error(), 0.0);
}

TEST(StatisticsTest, SingleValueHasZeroVariance) {
  RunningStats stats;

  stats.add(4.0);

  EXPECT_EQ(stats.count(), 1);
  EXPECT_DOUBLE_EQ(stats.mean(), 4.0);
  EXPECT_DOUBLE_EQ(stats.variance(), 0.0);
  EXPECT_DOUBLE_EQ(stats.standard_deviation(), 0.0);
  EXPECT_DOUBLE_EQ(stats.standard_error(), 0.0);
}

TEST(StatisticsTest, ComputesSampleStatistics) {
  RunningStats stats;

  stats.add(1.0);
  stats.add(2.0);
  stats.add(3.0);

  EXPECT_EQ(stats.count(), 3);
  EXPECT_DOUBLE_EQ(stats.mean(), 2.0);
  EXPECT_DOUBLE_EQ(stats.variance(), 1.0);
  EXPECT_DOUBLE_EQ(stats.standard_deviation(), 1.0);
  EXPECT_DOUBLE_EQ(stats.standard_error(), 1.0 / std::sqrt(3.0));
}

TEST(StatisticsTest, RejectsNonfiniteValues) {
  RunningStats stats;

  EXPECT_THROW(stats.add(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
  EXPECT_THROW(stats.add(std::numeric_limits<double>::infinity()), std::invalid_argument);
}

}  // namespace
