#include "vmc/statistics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

using vmc::BinningStats;
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

TEST(StatisticsTest, BinningRejectsZeroBinSize) {
  EXPECT_THROW(BinningStats{0}, std::invalid_argument);
}

TEST(StatisticsTest, BinningTracksRawStatsForEverySample) {
  BinningStats stats{3};

  stats.add(1.0);
  stats.add(2.0);

  EXPECT_EQ(stats.bin_size(), 3);
  EXPECT_EQ(stats.sample_count(), 2);
  EXPECT_EQ(stats.completed_bin_count(), 0);
  EXPECT_EQ(stats.current_bin_count(), 2);
  EXPECT_EQ(stats.raw_stats().count(), 2);
  EXPECT_DOUBLE_EQ(stats.raw_stats().mean(), 1.5);
  EXPECT_EQ(stats.bin_stats().count(), 0);
}

TEST(StatisticsTest, BinningAddsCompletedBinMeans) {
  BinningStats stats{2};

  stats.add(1.0);
  stats.add(3.0);
  stats.add(5.0);
  stats.add(7.0);

  EXPECT_EQ(stats.sample_count(), 4);
  EXPECT_EQ(stats.completed_bin_count(), 2);
  EXPECT_EQ(stats.current_bin_count(), 0);
  EXPECT_EQ(stats.raw_stats().count(), 4);
  EXPECT_DOUBLE_EQ(stats.raw_stats().mean(), 4.0);
  EXPECT_EQ(stats.bin_stats().count(), 2);
  EXPECT_DOUBLE_EQ(stats.bin_stats().mean(), 4.0);
  EXPECT_DOUBLE_EQ(stats.bin_stats().variance(), 8.0);
}

TEST(StatisticsTest, BinningIgnoresIncompleteTrailingBinForBinStats) {
  BinningStats stats{2};

  stats.add(1.0);
  stats.add(3.0);
  stats.add(100.0);

  EXPECT_EQ(stats.sample_count(), 3);
  EXPECT_EQ(stats.completed_bin_count(), 1);
  EXPECT_EQ(stats.current_bin_count(), 1);
  EXPECT_EQ(stats.bin_stats().count(), 1);
  EXPECT_DOUBLE_EQ(stats.bin_stats().mean(), 2.0);
  EXPECT_DOUBLE_EQ(stats.raw_stats().mean(), 104.0 / 3.0);
}

TEST(StatisticsTest, BinningRejectsNonfiniteValues) {
  BinningStats stats{2};

  EXPECT_THROW(stats.add(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
  EXPECT_THROW(stats.add(std::numeric_limits<double>::infinity()), std::invalid_argument);
}

}  // namespace
