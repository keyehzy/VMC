#include "vmc/wave_function.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

using vmc::BosonHop;
using vmc::BosonState;
using vmc::CondensateWaveFunction;
using vmc::OccupancyConstraint;

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

  EXPECT_DOUBLE_EQ(wave_function.ratio(state, hop), std::exp(wave_function.log_ratio(state, hop)));
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

}  // namespace
