#include "vmc/sr_optimizer.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using testing::ElementsAre;
using vmc::SrAccumulator;
using vmc::SrUpdateConfig;

std::vector<double> values_of(const Eigen::VectorXd& values) {
  return {values.data(), values.data() + values.size()};
}

MATCHER_P(DoubleNear, expected, "") {
  return testing::ExplainMatchResult(testing::DoubleNear(expected, 1.0e-12), arg, result_listener);
}

SrAccumulator rank_deficient_statistics() {
  SrAccumulator statistics{2};
  Eigen::VectorXd first(2);
  first << 2.0, -1.0;
  Eigen::VectorXd second(2);
  second << 4.0, 1.0;
  Eigen::VectorXd third(2);
  third << 6.0, 3.0;

  statistics.add(1.0, first);
  statistics.add(3.0, second);
  statistics.add(5.0, third);

  return statistics;
}

TEST(SrOptimizerTest, ComputesRegularizedUpdate) {
  const SrAccumulator statistics = rank_deficient_statistics();

  const auto result = compute_sr_update(statistics, SrUpdateConfig{
                                                        .step_size = 0.5,
                                                        .diagonal_shift = 1.0,
                                                    });

  Eigen::MatrixXd expected_covariance(2, 2);
  expected_covariance << 8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0;
  Eigen::MatrixXd expected_regularized(2, 2);
  expected_regularized << 11.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0, 11.0 / 3.0;

  EXPECT_TRUE(result.covariance.isApprox(expected_covariance));
  EXPECT_TRUE(result.regularized_covariance.isApprox(expected_regularized));
  EXPECT_THAT(values_of(result.forces),
              ElementsAre(DoubleNear(-16.0 / 3.0), DoubleNear(-16.0 / 3.0)));
  EXPECT_THAT(values_of(result.delta),
              ElementsAre(DoubleNear(-8.0 / 19.0), DoubleNear(-8.0 / 19.0)));
}

TEST(SrOptimizerTest, RejectsZeroSampleStatistics) {
  const SrAccumulator statistics{2};

  EXPECT_THROW((void)compute_sr_update(statistics,
                                       SrUpdateConfig{
                                           .step_size = 0.5,
                                           .diagonal_shift = 1.0,
                                       }),
               std::invalid_argument);
}

TEST(SrOptimizerTest, RejectsInvalidConfig) {
  const SrAccumulator statistics = rank_deficient_statistics();

  EXPECT_THROW((void)compute_sr_update(statistics,
                                       SrUpdateConfig{
                                           .step_size = 0.0,
                                           .diagonal_shift = 1.0,
                                       }),
               std::invalid_argument);
  EXPECT_THROW((void)compute_sr_update(statistics,
                                       SrUpdateConfig{
                                           .step_size = std::numeric_limits<double>::infinity(),
                                           .diagonal_shift = 1.0,
                                       }),
               std::invalid_argument);
  EXPECT_THROW((void)compute_sr_update(statistics,
                                       SrUpdateConfig{
                                           .step_size = 0.5,
                                           .diagonal_shift = -1.0,
                                       }),
               std::invalid_argument);
  EXPECT_THROW(
      (void)compute_sr_update(statistics,
                              SrUpdateConfig{
                                  .step_size = 0.5,
                                  .diagonal_shift = 1.0,
                                  .singular_tolerance = std::numeric_limits<double>::quiet_NaN(),
                              }),
      std::invalid_argument);
}

TEST(SrOptimizerTest, RejectsSingularUnregularizedCovariance) {
  const SrAccumulator statistics = rank_deficient_statistics();

  EXPECT_THROW((void)compute_sr_update(statistics,
                                       SrUpdateConfig{
                                           .step_size = 0.5,
                                           .diagonal_shift = 0.0,
                                       }),
               std::runtime_error);
}

TEST(SrOptimizerTest, SolvesSingleSampleWithDiagonalShift) {
  SrAccumulator statistics{2};
  Eigen::VectorXd derivatives(2);
  derivatives << -0.5, 0.0;
  statistics.add(-2.0, derivatives);

  const auto result = compute_sr_update(statistics, SrUpdateConfig{
                                                        .step_size = 0.25,
                                                        .diagonal_shift = 2.0,
                                                    });

  EXPECT_TRUE(result.covariance.isZero());
  EXPECT_TRUE(result.forces.isZero());
  EXPECT_TRUE(result.delta.isZero());
  EXPECT_DOUBLE_EQ(result.regularized_covariance(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(result.regularized_covariance(1, 1), 2.0);
}

}  // namespace
