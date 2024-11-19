#pragma once

#include "Brackets.hpp"
#include "Exporter.hpp"
#include "Filter.hpp"
#include "HermiteRunner.hpp"
#include "Transformer.hpp"
#include "constants.hpp"
#include "debug.hpp"
#include "grid.hpp"
#include "hyper.hpp"
#include "nonlinears.hpp"

#include <fftw-cpp/fftw-cpp.h>
#include <spdlog/spdlog.h>

#include <iomanip>
#include <type_traits>

namespace ahr {
namespace stdex = std::experimental;

class Naive : public ahr::HermiteRunner {
public:
  /**
   * Construct a new simulation using the Naive approach.
   * @param M The number of moments. Note that moments are 0-indexed, meaning that the highest
   * moment is M-1.
   * @param X The size of the X domain.
   * @param Y The size of the Y domain.
   */
  Naive(Dim M, Dim X, Dim Y);

  void init(std::string_view equilibriumName) override;

  void run(Dim N, Dim saveInterval) override;

  Grid g;
  Transformer tf{g};
  Exporter exporter{g, tf};
  HouLiFilter hlFilter{g};
  Brackets br{g, tf, hlFilter};

private:

  using View = Grid::View;
  using Buf = Grid::Buf;

  const Real XYNorm{1.0 / Real(g.X) / Real(g.Y)}; ///< Normalization factor for FFT

  Real dt{-1};        ///< timestep
  Real elapsedT{0.0}; ///< total time elapsed

  void fftHL(View::R_XY in, View::C_XY out); ///< FFT with Hou-Li Filter

  Real bPerpMax{0};

  static constexpr Dim N_E = 0;
  static constexpr Dim A_PAR = 1;
  static constexpr Dim G_MIN = 2;
  const Dim LAST = g.M - 1; ///< This is equivalent to ngtot in Viriato
  template <class T> using DxDy = Grid::DxDy<T>;

  /// \defgroup Buffers for all the physical quantities used.
  /// Names ending in K mean the values are in phase space.
  /// @{

  /// g_m: moment values for moments $m \in [0,M-1]$.
  /// The following values are special moments:
  /// - m=0: n_e (charge density)
  /// - m=1: A∥ (or Apar, parallel velocity)
  /// TODO maybe instead of these enormous amounts of memory, we could reuse (parallelism might
  /// suffer)
  Buf::C_MXY moments_K, momentsNew_K;

  /// A|| equilibrium value, used in corrector step
  Buf::C_XY aParEq_K;

  /// Φ: the electrostatic potential.
  Buf::C_XY phi_K, phi_K_New;

  /// sq(∇⊥) A∥, also parallel electron velocity
  Buf::C_XY ueKPar_K, ueKPar_K_New;

  /// Derivatives of moments
  DxDy<Buf::R_MXY> dGM;

  /// Derivatives of phi and ueKPar
  DxDy<Buf::R_XY> dPhi, dUEKPar;

  /// @}

  View::C_XY momentK(Dim m) { return g.sliceXY(moments_K, m); }

  // =================
  // Math helpers
  // TODO other file/class
  // =================

  [[nodiscard]] Real exp_nu(Dim kx, Dim ky, Real nu2, Real dt) const {
    return std::exp(-(nu * g.kPerp2(kx, ky) + nu2 * std::pow(g.kPerp2(kx, ky), hyper_order)) * dt);
  }

  [[nodiscard]] Real exp_gm(Dim m, Real hyper_nuei, Real dt) const {
    return std::exp(-(Real(m) * nu_ei + std::pow(m, 2 * hyper_morder) * hyper_nuei) * dt);
  }

  [[nodiscard]] Real exp_eta(Dim kx, Dim ky, Real res2, Real dt) const {
    return std::exp(-(res * g.kPerp2(kx, ky) + res2 * std::pow(g.kPerp2(kx, ky), hyper_order)) *
                    dt / (1.0 + g.kPerp2(kx, ky) * de * de));
  }

  /// getTimestep calculates flows and magnetic fields to determine a dt.
  /// It also updates bPerpMax in the process.
  [[nodiscard]] Real getTimestep(DxDy<View::R_XY> dPhi, DxDy<View::R_XY> dNE,
                                 DxDy<View::R_XY> dAPar);

public:
  struct Energies {
    Real magnetic{0.0}, kinetic{0.0};

    Real total() const { return magnetic + kinetic; }
  };

  Energies calculateEnergies() const;

  Real elapsedTime() const { return elapsedT; }

  Buf::R_XY getMoment(Dim m) const;

private:
  Real updateTimestep(Real dt, Real tempDt, bool noInc, Real relative_error) const;

  void exportTimestep(Dim t);
};
}; // namespace ahr