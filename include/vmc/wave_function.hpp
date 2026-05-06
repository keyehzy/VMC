#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "vmc/boson_state.hpp"
#include "vmc/proposal.hpp"

namespace vmc {

class WaveFunction {
 public:
  virtual ~WaveFunction() = default;

  [[nodiscard]] virtual std::size_t site_count() const = 0;
  [[nodiscard]] virtual double log_amplitude(const BosonState& state) const = 0;
  [[nodiscard]] virtual double log_ratio(const BosonState& state, const BosonHop& hop) const = 0;

  [[nodiscard]] double ratio(const BosonState& state, const BosonHop& hop) const;
};

class CondensateWaveFunction final : public WaveFunction {
 public:
  explicit CondensateWaveFunction(Eigen::VectorXd orbital);

  static CondensateWaveFunction uniform(std::size_t site_count);

  [[nodiscard]] std::size_t site_count() const override;

  [[nodiscard]] double log_amplitude(const BosonState& state) const override;
  [[nodiscard]] double log_ratio(const BosonState& state, const BosonHop& hop) const override;

 private:
  Eigen::VectorXd orbital_;
};

class JastrowWaveFunction final : public WaveFunction {
 public:
  explicit JastrowWaveFunction(Eigen::MatrixXd parameters);

  static JastrowWaveFunction zero(std::size_t site_count);

  [[nodiscard]] std::size_t site_count() const override;

  [[nodiscard]] double log_amplitude(const BosonState& state) const override;
  [[nodiscard]] double log_ratio(const BosonState& state, const BosonHop& hop) const override;

  [[nodiscard]] const Eigen::MatrixXd& parameters() const;

 private:
  Eigen::MatrixXd parameters_;
};

class JastrowCache {
 public:
  JastrowCache(const JastrowWaveFunction& wave_function, const BosonState& state);

  [[nodiscard]] std::size_t site_count() const;
  [[nodiscard]] std::span<const double> values() const;

  [[nodiscard]] double log_ratio(const BosonState& state, const BosonHop& hop) const;
  [[nodiscard]] double ratio(const BosonState& state, const BosonHop& hop) const;

  void apply_accepted_hop(const BosonHop& hop);

 private:
  std::reference_wrapper<const JastrowWaveFunction> wave_function_;
  std::vector<double> values_;
};

class ProductWaveFunction final : public WaveFunction {
 public:
  explicit ProductWaveFunction(std::vector<std::reference_wrapper<const WaveFunction>> components);

  [[nodiscard]] std::size_t site_count() const override;

  [[nodiscard]] double log_amplitude(const BosonState& state) const override;
  [[nodiscard]] double log_ratio(const BosonState& state, const BosonHop& hop) const override;

  [[nodiscard]] std::size_t component_count() const;

 private:
  std::vector<std::reference_wrapper<const WaveFunction>> components_;
  std::size_t site_count_;
};

}  // namespace vmc
