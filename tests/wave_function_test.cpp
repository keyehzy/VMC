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
using vmc::JastrowWaveFunction;
using vmc::OccupancyConstraint;
using vmc::WaveFunction;

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

}  // namespace
