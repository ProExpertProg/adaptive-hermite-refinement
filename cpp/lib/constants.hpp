#pragma once

#include "typedefs.hpp"

namespace ahr {
using std::numbers::pi;
/// Box
constexpr Real lx = 1.0 * 2 * pi;
constexpr Real ly = 1.0 * 2 * pi;

// TODO organize these into struct instead of global

/// Time parameters
constexpr Real InitAA0Fac = 0.1;
inline Real CFLFrac = 0.2;
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
inline Real res = 0.1;
inline Real nu = 0.1; // This is niu in Viriato
constexpr Real hyper_coef_g = 0.0;
constexpr Real hyper_coef = 0.0;
constexpr Real hyperm_coef = 0.0;

constexpr Dim hyper_order = 3;
constexpr Dim hyper_order_g = 3;
constexpr Dim hyper_morder = 3;

/// equil
constexpr Real a0 = 1.0;
} // namespace ahr