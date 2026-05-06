#include "vmc/wave_function.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using vmc::BosonHop;
using vmc::BosonState;
using vmc::CachedProductWaveFunction;
using vmc::CondensateWaveFunction;
using vmc::JastrowCache;
using vmc::JastrowWaveFunction;
using vmc::OccupancyConstraint;
using vmc::ProductWaveFunction;
using vmc::WaveFunction;

std::vector<double> cache_values(const JastrowCache& cache) {
  const auto values = cache.values();
  return {values.begin(), values.end()};
}

void expect_cache_values_near(const JastrowCache& lhs, const JastrowCache& rhs) {
  ASSERT_EQ(lhs.values().size(), rhs.values().size());
  for (std::size_t i = 0; i < lhs.values().size(); ++i) {
    EXPECT_NEAR(lhs.values()[i], rhs.values()[i], 1.0e-12);
  }
}

TEST(WaveFunctionTest, BuildsUniformCondensate) {
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(4);

  EXPECT_EQ(wave_function.site_count(), 4);
}

TEST(WaveFunctionTest, ComputesUniformSoftCoreRatioFromOccupations) {
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(3);
  const BosonState state = BosonState::from_occupations({2, 1, 0}, OccupancyConstraint::SoftCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 2,
  };

  EXPECT_DOUBLE_EQ(wave_function.ratio(state, hop), std::sqrt(2.0));
}

TEST(WaveFunctionTest, ComputesNonUniformSoftCoreRatioFromOrbitalAndOccupations) {
  Eigen::VectorXd orbital(3);
  orbital << 0.5, 1.0, 2.0;
  const CondensateWaveFunction wave_function{orbital};
  const BosonState state = BosonState::from_occupations({2, 1, 1}, OccupancyConstraint::SoftCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 2,
  };

  const double expected = (2.0 / 0.5) * std::sqrt(2.0 / 2.0);

  EXPECT_DOUBLE_EQ(wave_function.ratio(state, hop), expected);
}

TEST(WaveFunctionTest, RatioIsExponentiatedLogRatio) {
  Eigen::VectorXd orbital(3);
  orbital << 0.5, 1.0, 2.0;
  const CondensateWaveFunction wave_function{orbital};
  const BosonState state = BosonState::from_occupations({2, 1, 1}, OccupancyConstraint::SoftCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 2,
  };

  const WaveFunction& base = wave_function;

  EXPECT_DOUBLE_EQ(base.ratio(state, hop), std::exp(base.log_ratio(state, hop)));
}

TEST(WaveFunctionTest, ComputesLogAmplitude) {
  Eigen::VectorXd orbital(3);
  orbital << 0.5, 1.0, 2.0;
  const CondensateWaveFunction wave_function{orbital};
  const BosonState state = BosonState::from_occupations({2, 1, 0}, OccupancyConstraint::SoftCore);

  const double expected = 2.0 * std::log(0.5) + std::log(1.0) - 0.5 * std::lgamma(3.0) -
                          0.5 * std::lgamma(2.0) - 0.5 * std::lgamma(1.0);

  EXPECT_DOUBLE_EQ(wave_function.log_amplitude(state), expected);
}

TEST(WaveFunctionTest, RejectsInvalidOrbitals) {
  EXPECT_THROW(CondensateWaveFunction{Eigen::VectorXd{}}, std::invalid_argument);

  Eigen::VectorXd zero(2);
  zero << 1.0, 0.0;
  EXPECT_THROW(CondensateWaveFunction{zero}, std::invalid_argument);

  Eigen::VectorXd negative(2);
  negative << 1.0, -1.0;
  EXPECT_THROW(CondensateWaveFunction{negative}, std::invalid_argument);

  Eigen::VectorXd infinite(2);
  infinite << 1.0, std::numeric_limits<double>::infinity();
  EXPECT_THROW(CondensateWaveFunction{infinite}, std::invalid_argument);

  EXPECT_THROW(CondensateWaveFunction::uniform(0), std::invalid_argument);
}

TEST(WaveFunctionTest, RejectsStateSizeMismatch) {
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(3);
  const BosonState state = BosonState::from_occupations({1, 0}, OccupancyConstraint::SoftCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };

  EXPECT_THROW((void)wave_function.log_amplitude(state), std::invalid_argument);
  EXPECT_THROW((void)wave_function.log_ratio(state, hop), std::invalid_argument);
}

TEST(WaveFunctionTest, RejectsHopWhoseSourceDoesNotMatchSelectedBoson) {
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(3);
  const BosonState state =
      BosonState::from_boson_positions(3, {0, 1}, OccupancyConstraint::SoftCore);
  const BosonHop hop{
      .boson = 0,
      .source = 1,
      .destination = 2,
  };

  EXPECT_THROW((void)wave_function.log_ratio(state, hop), std::invalid_argument);
}

TEST(WaveFunctionTest, RejectsTrivialHop) {
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(3);
  const BosonState state = BosonState::from_boson_positions(3, {0}, OccupancyConstraint::SoftCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 0,
  };

  EXPECT_THROW((void)wave_function.log_ratio(state, hop), std::invalid_argument);
}

TEST(WaveFunctionTest, RejectsHardCoreOccupiedDestination) {
  const CondensateWaveFunction wave_function = CondensateWaveFunction::uniform(3);
  const BosonState state =
      BosonState::from_boson_positions(3, {0, 1}, OccupancyConstraint::HardCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };

  EXPECT_THROW((void)wave_function.log_ratio(state, hop), std::invalid_argument);
}

TEST(WaveFunctionTest, BuildsZeroJastrow) {
  const JastrowWaveFunction wave_function = JastrowWaveFunction::zero(3);

  EXPECT_EQ(wave_function.site_count(), 3);
  EXPECT_EQ(wave_function.parameters().rows(), 3);
  EXPECT_EQ(wave_function.parameters().cols(), 3);
  EXPECT_TRUE(wave_function.parameters().isZero());
}

TEST(WaveFunctionTest, ZeroJastrowHasUnitRatioAndZeroLogAmplitude) {
  const JastrowWaveFunction wave_function = JastrowWaveFunction::zero(3);
  const BosonState state = BosonState::from_occupations({2, 1, 0}, OccupancyConstraint::SoftCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 2,
  };

  EXPECT_DOUBLE_EQ(wave_function.log_amplitude(state), 0.0);
  EXPECT_DOUBLE_EQ(wave_function.log_ratio(state, hop), 0.0);
  EXPECT_DOUBLE_EQ(wave_function.ratio(state, hop), 1.0);
}

TEST(WaveFunctionTest, ComputesDiagonalJastrowLogAmplitude) {
  Eigen::MatrixXd parameters = Eigen::MatrixXd::Zero(2, 2);
  parameters(0, 0) = 0.25;
  parameters(1, 1) = 0.5;
  const JastrowWaveFunction wave_function{parameters};
  const BosonState state = BosonState::from_occupations({2, 1}, OccupancyConstraint::SoftCore);

  EXPECT_DOUBLE_EQ(wave_function.log_amplitude(state), -0.75);
}

TEST(WaveFunctionTest, ComputesOffDiagonalJastrowLogAmplitude) {
  Eigen::MatrixXd parameters = Eigen::MatrixXd::Zero(2, 2);
  parameters(0, 1) = 0.25;
  parameters(1, 0) = 0.25;
  const JastrowWaveFunction wave_function{parameters};
  const BosonState state = BosonState::from_occupations({2, 3}, OccupancyConstraint::SoftCore);

  EXPECT_DOUBLE_EQ(wave_function.log_amplitude(state), -1.5);
}

TEST(WaveFunctionTest, ComputesJastrowRatioFromDirectAmplitudes) {
  Eigen::MatrixXd parameters(2, 2);
  parameters << 0.2, 0.1, 0.1, 0.4;
  const JastrowWaveFunction wave_function{parameters};
  const BosonState before = BosonState::from_occupations({2, 1}, OccupancyConstraint::SoftCore);
  const BosonState after = BosonState::from_occupations({1, 2}, OccupancyConstraint::SoftCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };

  const double expected = wave_function.log_amplitude(after) - wave_function.log_amplitude(before);

  EXPECT_DOUBLE_EQ(wave_function.log_ratio(before, hop), expected);
  EXPECT_DOUBLE_EQ(wave_function.ratio(before, hop), std::exp(expected));
}

TEST(WaveFunctionTest, RejectsInvalidJastrowParameters) {
  EXPECT_THROW(JastrowWaveFunction{Eigen::MatrixXd{}}, std::invalid_argument);
  EXPECT_THROW(JastrowWaveFunction{Eigen::MatrixXd::Zero(2, 3)}, std::invalid_argument);

  Eigen::MatrixXd nonsymmetric(2, 2);
  nonsymmetric << 1.0, 2.0, 3.0, 4.0;
  EXPECT_THROW(JastrowWaveFunction{nonsymmetric}, std::invalid_argument);

  Eigen::MatrixXd infinite = Eigen::MatrixXd::Zero(2, 2);
  infinite(0, 1) = std::numeric_limits<double>::infinity();
  infinite(1, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW(JastrowWaveFunction{infinite}, std::invalid_argument);

  EXPECT_THROW(JastrowWaveFunction::zero(0), std::invalid_argument);
}

TEST(WaveFunctionTest, JastrowRejectsStateSizeMismatch) {
  const JastrowWaveFunction wave_function = JastrowWaveFunction::zero(3);
  const BosonState state = BosonState::from_occupations({1, 0}, OccupancyConstraint::SoftCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };

  EXPECT_THROW((void)wave_function.log_amplitude(state), std::invalid_argument);
  EXPECT_THROW((void)wave_function.log_ratio(state, hop), std::invalid_argument);
}

TEST(WaveFunctionTest, JastrowRejectsHardCoreOccupiedDestination) {
  const JastrowWaveFunction wave_function = JastrowWaveFunction::zero(3);
  const BosonState state =
      BosonState::from_boson_positions(3, {0, 1}, OccupancyConstraint::HardCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };

  EXPECT_THROW((void)wave_function.log_ratio(state, hop), std::invalid_argument);
}

TEST(WaveFunctionTest, JastrowCacheBuildsInitialValues) {
  Eigen::MatrixXd parameters(3, 3);
  parameters << 0.2, 0.1, -0.3, 0.1, 0.4, 0.5, -0.3, 0.5, 0.7;
  const JastrowWaveFunction wave_function{parameters};
  const BosonState state = BosonState::from_occupations({2, 0, 1}, OccupancyConstraint::SoftCore);

  const JastrowCache cache{wave_function, state};

  EXPECT_EQ(cache.site_count(), 3);
  EXPECT_THAT(cache_values(cache), testing::ElementsAre(testing::DoubleNear(0.1, 1.0e-12),
                                                        testing::DoubleNear(0.7, 1.0e-12),
                                                        testing::DoubleNear(0.1, 1.0e-12)));
}

TEST(WaveFunctionTest, JastrowCacheMatchesDirectLogRatio) {
  Eigen::MatrixXd parameters(3, 3);
  parameters << 0.2, 0.1, -0.3, 0.1, 0.4, 0.5, -0.3, 0.5, 0.7;
  const JastrowWaveFunction wave_function{parameters};
  const BosonState state = BosonState::from_occupations({2, 0, 1}, OccupancyConstraint::SoftCore);
  const JastrowCache cache{wave_function, state};
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };

  EXPECT_DOUBLE_EQ(cache.log_ratio(state, hop), wave_function.log_ratio(state, hop));
  EXPECT_DOUBLE_EQ(cache.ratio(state, hop), std::exp(cache.log_ratio(state, hop)));
}

TEST(WaveFunctionTest, JastrowCacheUpdateMatchesFreshCacheAfterAcceptedHop) {
  Eigen::MatrixXd parameters(3, 3);
  parameters << 0.2, 0.1, -0.3, 0.1, 0.4, 0.5, -0.3, 0.5, 0.7;
  const JastrowWaveFunction wave_function{parameters};
  BosonState state = BosonState::from_occupations({2, 0, 1}, OccupancyConstraint::SoftCore);
  JastrowCache cache{wave_function, state};
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };

  state.move_boson(hop.boson, hop.destination);
  cache.apply_accepted_hop(hop);
  const JastrowCache fresh_cache{wave_function, state};

  expect_cache_values_near(cache, fresh_cache);
}

TEST(WaveFunctionTest, JastrowCacheStaysConsistentAcrossMultipleAcceptedHops) {
  Eigen::MatrixXd parameters(3, 3);
  parameters << 0.2, 0.1, -0.3, 0.1, 0.4, 0.5, -0.3, 0.5, 0.7;
  const JastrowWaveFunction wave_function{parameters};
  BosonState state = BosonState::from_occupations({2, 0, 1}, OccupancyConstraint::SoftCore);
  JastrowCache cache{wave_function, state};
  const BosonHop first_hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };
  const BosonHop second_hop{
      .boson = 2,
      .source = 2,
      .destination = 1,
  };

  state.move_boson(first_hop.boson, first_hop.destination);
  cache.apply_accepted_hop(first_hop);
  state.move_boson(second_hop.boson, second_hop.destination);
  cache.apply_accepted_hop(second_hop);
  const JastrowCache fresh_cache{wave_function, state};

  expect_cache_values_near(cache, fresh_cache);
}

TEST(WaveFunctionTest, JastrowCacheRejectsStateSizeMismatch) {
  const JastrowWaveFunction wave_function = JastrowWaveFunction::zero(3);
  const BosonState state = BosonState::from_occupations({1, 0}, OccupancyConstraint::SoftCore);

  EXPECT_THROW(JastrowCache(wave_function, state), std::invalid_argument);
}

TEST(WaveFunctionTest, JastrowCacheRejectsHardCoreOccupiedDestination) {
  const JastrowWaveFunction wave_function = JastrowWaveFunction::zero(2);
  const BosonState state =
      BosonState::from_boson_positions(2, {0, 1}, OccupancyConstraint::HardCore);
  const JastrowCache cache{wave_function, state};
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };

  EXPECT_THROW((void)cache.log_ratio(state, hop), std::invalid_argument);
}

TEST(WaveFunctionTest, JastrowCacheRejectsInvalidUpdateHop) {
  const JastrowWaveFunction wave_function = JastrowWaveFunction::zero(2);
  const BosonState state = BosonState::from_boson_positions(2, {0}, OccupancyConstraint::SoftCore);
  JastrowCache cache{wave_function, state};

  EXPECT_THROW(cache.apply_accepted_hop(BosonHop{.boson = 0, .source = 0, .destination = 0}),
               std::invalid_argument);
  EXPECT_THROW(cache.apply_accepted_hop(BosonHop{.boson = 0, .source = 0, .destination = 2}),
               std::out_of_range);
}

TEST(WaveFunctionTest, BuildsProductWaveFunction) {
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(3);
  const JastrowWaveFunction jastrow = JastrowWaveFunction::zero(3);
  const ProductWaveFunction product{{condensate, jastrow}};

  EXPECT_EQ(product.site_count(), 3);
  EXPECT_EQ(product.component_count(), 2);
}

TEST(WaveFunctionTest, ProductRejectsEmptyComponents) {
  EXPECT_THROW(ProductWaveFunction{{}}, std::invalid_argument);
}

TEST(WaveFunctionTest, ProductRejectsMismatchedComponentSizes) {
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(3);
  const JastrowWaveFunction jastrow = JastrowWaveFunction::zero(4);

  EXPECT_THROW(ProductWaveFunction({condensate, jastrow}), std::invalid_argument);
}

TEST(WaveFunctionTest, ProductLogAmplitudeIsSumOfComponentLogAmplitudes) {
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(2);
  Eigen::MatrixXd parameters(2, 2);
  parameters << 0.2, 0.1, 0.1, 0.4;
  const JastrowWaveFunction jastrow{parameters};
  const ProductWaveFunction product{{condensate, jastrow}};
  const BosonState state = BosonState::from_occupations({2, 1}, OccupancyConstraint::SoftCore);

  const double expected = condensate.log_amplitude(state) + jastrow.log_amplitude(state);

  EXPECT_DOUBLE_EQ(product.log_amplitude(state), expected);
}

TEST(WaveFunctionTest, ProductLogRatioIsSumOfComponentLogRatios) {
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(2);
  Eigen::MatrixXd parameters(2, 2);
  parameters << 0.2, 0.1, 0.1, 0.4;
  const JastrowWaveFunction jastrow{parameters};
  const ProductWaveFunction product{{condensate, jastrow}};
  const BosonState state = BosonState::from_occupations({2, 1}, OccupancyConstraint::SoftCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };

  const double expected = condensate.log_ratio(state, hop) + jastrow.log_ratio(state, hop);

  EXPECT_DOUBLE_EQ(product.log_ratio(state, hop), expected);
}

TEST(WaveFunctionTest, ProductRatioUsesBaseExponentiatedLogRatio) {
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(2);
  Eigen::MatrixXd parameters(2, 2);
  parameters << 0.2, 0.1, 0.1, 0.4;
  const JastrowWaveFunction jastrow{parameters};
  const ProductWaveFunction product{{condensate, jastrow}};
  const BosonState state = BosonState::from_occupations({2, 1}, OccupancyConstraint::SoftCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };
  const WaveFunction& base = product;

  EXPECT_DOUBLE_EQ(base.ratio(state, hop), std::exp(base.log_ratio(state, hop)));
}

TEST(WaveFunctionTest, ProductWithZeroJastrowMatchesCondensate) {
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(3);
  const JastrowWaveFunction jastrow = JastrowWaveFunction::zero(3);
  const ProductWaveFunction product{{condensate, jastrow}};
  const BosonState state = BosonState::from_occupations({2, 1, 0}, OccupancyConstraint::SoftCore);
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 2,
  };

  EXPECT_DOUBLE_EQ(product.log_amplitude(state), condensate.log_amplitude(state));
  EXPECT_DOUBLE_EQ(product.log_ratio(state, hop), condensate.log_ratio(state, hop));
  EXPECT_DOUBLE_EQ(product.ratio(state, hop), condensate.ratio(state, hop));
}

TEST(WaveFunctionTest, CachedProductLogRatioMatchesDirectProduct) {
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(3);
  Eigen::MatrixXd parameters(3, 3);
  parameters << 0.2, 0.1, -0.3, 0.1, 0.4, 0.5, -0.3, 0.5, 0.7;
  const JastrowWaveFunction jastrow{parameters};
  const ProductWaveFunction direct_product{{condensate, jastrow}};
  const BosonState state = BosonState::from_occupations({2, 0, 1}, OccupancyConstraint::SoftCore);
  const CachedProductWaveFunction cached_product{condensate, jastrow, state};
  const BosonHop hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };

  EXPECT_EQ(cached_product.site_count(), 3);
  EXPECT_DOUBLE_EQ(cached_product.log_ratio(state, hop), direct_product.log_ratio(state, hop));
  EXPECT_DOUBLE_EQ(cached_product.ratio(state, hop),
                   std::exp(cached_product.log_ratio(state, hop)));
}

TEST(WaveFunctionTest, CachedProductRejectsMismatchedInputs) {
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(3);
  const JastrowWaveFunction jastrow = JastrowWaveFunction::zero(4);
  const BosonState state = BosonState::from_occupations({1, 0, 0}, OccupancyConstraint::SoftCore);

  EXPECT_THROW(CachedProductWaveFunction(condensate, jastrow, state), std::invalid_argument);
}

TEST(WaveFunctionTest, CachedProductStaysConsistentAfterAcceptedHop) {
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(3);
  Eigen::MatrixXd parameters(3, 3);
  parameters << 0.2, 0.1, -0.3, 0.1, 0.4, 0.5, -0.3, 0.5, 0.7;
  const JastrowWaveFunction jastrow{parameters};
  BosonState state = BosonState::from_occupations({2, 0, 1}, OccupancyConstraint::SoftCore);
  CachedProductWaveFunction cached_product{condensate, jastrow, state};
  const BosonHop accepted_hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };

  state.move_boson(accepted_hop.boson, accepted_hop.destination);
  cached_product.apply_accepted_hop(accepted_hop);

  const ProductWaveFunction direct_product{{condensate, jastrow}};
  const BosonHop next_hop{
      .boson = 2,
      .source = 2,
      .destination = 0,
  };

  EXPECT_DOUBLE_EQ(cached_product.log_ratio(state, next_hop),
                   direct_product.log_ratio(state, next_hop));
}

TEST(WaveFunctionTest, CachedProductStaysConsistentAfterMultipleAcceptedHops) {
  const CondensateWaveFunction condensate = CondensateWaveFunction::uniform(3);
  Eigen::MatrixXd parameters(3, 3);
  parameters << 0.2, 0.1, -0.3, 0.1, 0.4, 0.5, -0.3, 0.5, 0.7;
  const JastrowWaveFunction jastrow{parameters};
  BosonState state = BosonState::from_occupations({2, 0, 1}, OccupancyConstraint::SoftCore);
  CachedProductWaveFunction cached_product{condensate, jastrow, state};
  const BosonHop first_hop{
      .boson = 0,
      .source = 0,
      .destination = 1,
  };
  const BosonHop second_hop{
      .boson = 2,
      .source = 2,
      .destination = 1,
  };

  state.move_boson(first_hop.boson, first_hop.destination);
  cached_product.apply_accepted_hop(first_hop);
  state.move_boson(second_hop.boson, second_hop.destination);
  cached_product.apply_accepted_hop(second_hop);

  const ProductWaveFunction direct_product{{condensate, jastrow}};
  const BosonHop next_hop{
      .boson = 1,
      .source = 0,
      .destination = 2,
  };

  EXPECT_DOUBLE_EQ(cached_product.log_ratio(state, next_hop),
                   direct_product.log_ratio(state, next_hop));
}

}  // namespace
