#include "vmc/hamiltonian.hpp"

#include <cmath>
#include <stdexcept>

namespace vmc {
namespace {

void require_compatible_sizes(const Lattice& lattice, const BosonState& state,
                              const WaveFunction& wave_function) {
  if (lattice.site_count() != state.site_count()) {
    throw std::invalid_argument("lattice and boson state must have the same site count");
  }

  if (wave_function.site_count() != state.site_count()) {
    throw std::invalid_argument("wave function and boson state must have the same site count");
  }
}

void require_finite_parameters(const BoseHubbardHamiltonian& hamiltonian) {
  if (!std::isfinite(hamiltonian.hopping_t) || !std::isfinite(hamiltonian.interaction_u)) {
    throw std::invalid_argument("Bose-Hubbard Hamiltonian parameters must be finite");
  }
}

double potential_energy(const BosonState& state, const BoseHubbardHamiltonian& hamiltonian) {
  double energy = 0.0;
  for (BosonState::Site site = 0; site < state.site_count(); ++site) {
    const std::size_t occupation = state.occupation(site);
    energy += 0.5 * hamiltonian.interaction_u * static_cast<double>(occupation) *
              static_cast<double>(occupation - 1);
  }
  return energy;
}

double kinetic_energy(const Lattice& lattice, const BosonState& state,
                      const WaveFunction& wave_function,
                      const BoseHubbardHamiltonian& hamiltonian) {
  double energy = 0.0;

  for (BosonState::Site source = 0; source < state.site_count(); ++source) {
    const std::size_t source_occupation = state.occupation(source);
    if (source_occupation == 0) {
      continue;
    }

    const std::optional<BosonState::Boson> boson = state.first_boson_at(source);
    if (!boson.has_value()) {
      throw std::logic_error("occupied site has no corresponding boson position");
    }

    for (const BosonState::Site destination : lattice.neighbors(source)) {
      const std::size_t destination_occupation = state.occupation(destination);
      if (state.is_hardcore() && destination_occupation > 0) {
        continue;
      }

      const BosonHop hop{
          .boson = *boson,
          .source = source,
          .destination = destination,
      };
      const double matrix_element = std::sqrt(static_cast<double>(source_occupation) *
                                              static_cast<double>(destination_occupation + 1));

      energy += -hamiltonian.hopping_t * matrix_element * wave_function.ratio(state, hop);
    }
  }

  return energy;
}

}  // namespace

double local_energy(const Lattice& lattice, const BosonState& state,
                    const WaveFunction& wave_function, const BoseHubbardHamiltonian& hamiltonian) {
  require_compatible_sizes(lattice, state, wave_function);
  require_finite_parameters(hamiltonian);

  return potential_energy(state, hamiltonian) +
         kinetic_energy(lattice, state, wave_function, hamiltonian);
}

}  // namespace vmc
