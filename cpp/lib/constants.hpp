#pragma once

#include "grid.hpp"
#include "typedefs.hpp"

namespace ahr {
using std::numbers::pi;
/// Box
constexpr Real lx = 1.0 * 2 * pi;
constexpr Real ly = 1.0 * 2 * pi;

// TODO organize these into struct instead of global

/// Time parameters
constexpr Real InitAA0Fac = 0.1;
constexpr Real CFLFrac = 0.05;
constexpr Real epsilon = 1e-10;

// This actually means MaxP+1 is the last computed value
constexpr Dim MaxP = 1;

constexpr Real aa0 = InitAA0Fac; /// TODO not const
constexpr Real low = 0.92;

/// FLR
constexpr Real rhoI = 1.0e-7;
constexpr Real rhoS = 1.0e-7;
constexpr Real de = 1.0e-7;

/// MHD
constexpr Real smallRhoI = 1.0e-6;

/// Diffusion
constexpr Real nu_ei = 0.0001;
constexpr Real res = 0.1;
constexpr Real nu = 0.1; // This is niu in Viriato
constexpr Real hyper_coef_g = 0.0;
constexpr Real hyper_coef = 0.0;
constexpr Real hyperm_coef = 0.0;

constexpr Dim hyper_order = 3;
constexpr Dim hyper_order_g = 3;
constexpr Dim hyper_morder = 3;

/// equil
constexpr Real a0 = 1.0;

struct HyperCoefficients {
  Real nu_g, nu_2, eta2, nu_ei;

  static HyperCoefficients calculate(Real dt, Grid const &g) {
    Real kPerpMax2 = std::pow(g.KX, 2) + std::pow(Real(g.KY) / 2, 2);

    HyperCoefficients ret{};
    ret.nu_g = hyper_coef_g / dt / std::pow(kPerpMax2, hyper_order_g);
    ret.nu_2 = hyper_coef / dt / std::pow(kPerpMax2, hyper_order);
    if (kPerpMax2 * de * de > 1) {
      ret.eta2 = hyper_coef / dt / std::pow(kPerpMax2, hyper_order - 1) * de * de;
    } else {
      ret.eta2 = hyper_coef / dt / std::pow(kPerpMax2, hyper_order);
    }

    ret.nu_ei = hyperm_coef / dt / std::pow(g.M, 2 * hyper_morder);

    return ret;
  }
};

} // namespace ahr