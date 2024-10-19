#pragma once

#include "typedefs.hpp"

namespace ahr {
using std::numbers::pi;
/// Box
constexpr Real lx = 1.0 * 2 * pi;
constexpr Real ly = 1.0 * 2 * pi;

// TODO organize these into struct instead of global

/// Time parameters
inline Real InitAA0Fac = 0.1;
inline Real CFLFrac = 0.2;
inline Real epsilon = 1e-10;

// This actually means MaxP+1 is the last computed value
inline Dim MaxP = 1;

inline Real aa0 = InitAA0Fac; /// TODO not const
inline Real low = 0.92;

/// FLR
inline Real rhoI = 1.0e-7;
inline Real rhoS = 1.0e-7;
inline Real de = 1.0e-7;

/// MHD
inline Real smallRhoI = 1.0e-6;

/// Diffusion
inline Real nu_ei = 0.0001;
inline Real res = 0.1;
inline Real nu = 0.1; // This is niu in Viriato
inline Real hyper_coef_g = 0.0;
inline Real hyper_coef = 0.0;
inline Real hyperm_coef = 0.0;

inline Dim hyper_order = 3;
inline Dim hyper_order_g = 3;
inline Dim hyper_morder = 3;

/// equil
inline Real a0 = 1.0;

struct HyperCoefficients {
  Real nu_g, nu_2, eta2, nu_ei;

  static HyperCoefficients calculate(Real dt, Dim KX, Dim KY, Dim M) {
    Real kPerpMax2 = std::pow(KX, 2) + std::pow(Real(KY) / 2, 2);

    HyperCoefficients ret{};
    ret.nu_g = hyper_coef_g / dt / std::pow(kPerpMax2, hyper_order_g);
    ret.nu_2 = hyper_coef / dt / std::pow(kPerpMax2, hyper_order);
    if (kPerpMax2 * de * de > 1) {
      ret.eta2 = hyper_coef / dt / std::pow(kPerpMax2, hyper_order - 1) * de * de;
    } else {
      ret.eta2 = hyper_coef / dt / std::pow(kPerpMax2, hyper_order);
    }

    ret.nu_ei = hyperm_coef / dt / std::pow(M, 2 * hyper_morder);

    return ret;
  }
};

} // namespace ahr