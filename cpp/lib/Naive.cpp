#include "Naive.hpp"
#include "equillibrium.hpp"

#include <cnpy.h>
#include <cstdlib>
#include <spdlog/fmt/bundled/ostream.h>
#include <utility>

namespace ahr {
Naive::Naive(Dim M, Dim X, Dim Y)
    : HermiteRunner(), g(M, X, Y),
      // TODO move to tracker class
      moments_K(g.cBufMXY()), momentsNew_K(g.cBufMXY()), phi_K(g.cBufXY()), phi_K_New(g.cBufXY()),
      ueKPar_K(g.cBufXY()), ueKPar_K_New(g.cBufXY()), aParEq_K(g.cBufXY()), dGM(g.dBufMXY()),
      dPhi(g.dBufXY()), dUEKPar(g.dBufXY()) {
  assert(M >= 4 or M == 2);
  // X and Y must be powers of 2
  assert((X & (X - 1)) == 0);
  assert((Y & (Y - 1)) == 0);
}

void Naive::fftHL(View::R_XY in, View::C_XY out) {
  tf.fft(in, out);
  hlFilter(out);
}

void Naive::init(std::string_view equilibriumName) {
  // Currently assuming X==Y for simplicity, but the code is written generally for the most part.
  assert(g.X == g.Y);
  auto temp = g.rBufXY();

  // Plan FFTs both ways
  tf.init();

  // Initialize equilibrium values
  auto [aParEq, phi] = equilibrium(equilibriumName, g);

  fftHL(phi, phi_K);
  fftHL(aParEq, aParEq_K);

  // Transform moments into phase space
  for (int m = G_MIN; m < g.M; ++m) {
    g.for_each_kxky([&](Dim kx, Dim ky) { moments_K(kx, ky, m) = 0; });
  }

  // aPar, uekPar, ne
  g.for_each_kxky([&](Dim kx, Dim ky) {
    moments_K(kx, ky, N_E) = nonlinear::phiInv(phi_K(kx, ky), g.kPerp2(kx, ky));
    moments_K(kx, ky, A_PAR) = aParEq_K(kx, ky);
    ueKPar_K(kx, ky) = -g.kPerp2(kx, ky) * moments_K(kx, ky, A_PAR);
  });
  br.derivatives(phi_K, dPhi);
  br.derivatives(ueKPar_K, dUEKPar);
  for (int m = 0; m < g.M; ++m) {
    br.derivatives(momentK(m), Grid::sliceXY(dGM, m));
  }
}

void Naive::run(Dim N, Dim saveInterval) {
  // isothermal if only running with 2 moments
  assert(g.M >= 4 or g.M == 2);

  bool divergent = false, repeat = false, noInc = false;
  int divergentCount = 0, repeatCount = 0;

  bool saved = false;
  // Manually increment t only if not diverging
  for (Dim t = 0; t < N;) {
    if (saveInterval != 0 and t % saveInterval == 0) {
      if (!saved) {
        spdlog::info("Saving for timestep: {}", t);
        saved = true;
        exportTimestep(t);
      }
    }

    // predictor step
    if (repeat or divergent) {
      spdlog::debug("repeat: {}, divergent: {}", repeat, divergent);
      repeat = false;
      divergent = false;
    } else if (dt == -1) {
      dt = getTimestep(dPhi, Grid::sliceXY(dGM, N_E), Grid::sliceXY(dGM, A_PAR));
    }

    hyper = HyperCoefficients::calculate(dt, g);
    exp_nu.update(hyper, dt);
    exp_nu_g.update(hyper, dt);
    exp_eta.update(hyper, dt);
    exp_gm.update(hyper, dt);

    spdlog::debug("dt: {}", dt);

    // store results of nonlinear operators, as well as results of predictor step
    auto GM_K_Star = g.cBufMXY(), GM_Nonlinear_K = g.cBufMXY();

    // Compute N
    auto bracketPhiNE_K = br.halfBracket(dPhi, Grid::sliceXY(dGM, N_E));
    auto bracketAParUEKPar_K = br.halfBracket(Grid::sliceXY(dGM, A_PAR), dUEKPar);

    // Compute A
    auto dPhiNeG2 = g.dBufXY();
    if (g.M > 2) {
      g.for_each_xy([&](Dim x, Dim y) {
        dPhiNeG2.DX(x, y) =
            dPhi.DX(x, y) - rhoS * rhoS * (std::sqrt(2) * dGM.DX(x, y, G_MIN) + dGM.DX(x, y, N_E));
        dPhiNeG2.DY(x, y) =
            dPhi.DY(x, y) - rhoS * rhoS * (std::sqrt(2) * dGM.DY(x, y, G_MIN) + dGM.DY(x, y, N_E));
      });
    } else {
      g.for_each_xy([&](Dim x, Dim y) {
        dPhiNeG2.DX(x, y) = dPhi.DX(x, y) - rhoS * rhoS * dGM.DX(x, y, N_E);
        dPhiNeG2.DY(x, y) = dPhi.DY(x, y) - rhoS * rhoS * dGM.DY(x, y, N_E);
      });
    }

    auto bracketAParPhiG2Ne_K = br.halfBracket(Grid::sliceXY(dGM, A_PAR), dPhiNeG2);
    auto bracketUEParPhi_K = br.halfBracket(dUEKPar, dPhi);

    g.for_each_kxky([&](Dim kx, Dim ky) {
      GM_Nonlinear_K(kx, ky, N_E) =
          nonlinear::N(bracketPhiNE_K(kx, ky), bracketAParUEKPar_K(kx, ky));
      GM_K_Star(kx, ky, N_E) = exp_nu(kx, ky) * moments_K(kx, ky, N_E) +
                               dt / 2.0 * (1 + exp_nu(kx, ky)) * GM_Nonlinear_K(kx, ky, N_E);

      GM_Nonlinear_K(kx, ky, A_PAR) =
          nonlinear::A(bracketAParPhiG2Ne_K(kx, ky), bracketUEParPhi_K(kx, ky), g.kPerp2(kx, ky));
      GM_K_Star(kx, ky, A_PAR) = exp_eta(kx, ky) * moments_K(kx, ky, A_PAR) +
                                 dt / 2.0 * (1 + exp_eta(kx, ky)) * GM_Nonlinear_K(kx, ky, A_PAR) +
                                 (1.0 - exp_eta(kx, ky)) * aParEq_K(kx, ky);
    });

    if (g.M > 2) {
      // Compute G2
      auto bracketPhiG2_K = br.halfBracket(dPhi, Grid::sliceXY(dGM, G_MIN));
      auto bracketAParG3_K =
          br.halfBracket(Grid::sliceXY(dGM, A_PAR), Grid::sliceXY(dGM, G_MIN + 1));

      // Compute G_{M-1}
      auto bracketPhiGLast_K = br.halfBracket(dPhi, Grid::sliceXY(dGM, LAST));
      auto bracketAParGLast_K = br.halfBracket(Grid::sliceXY(dGM, A_PAR), Grid::sliceXY(dGM, LAST));
      g.for_each_kxky([&](Dim kx, Dim ky) {
        bracketAParGLast_K(kx, ky) *= nonlinear::GLastBracketFactor(g.M, g.kPerp2(kx, ky), hyper);
        bracketAParGLast_K(kx, ky) += rhoS / de * std::sqrt(LAST) * moments_K(kx, ky, LAST - 1);
        // TODO Viriato adds this after the derivative
      });

      auto dBrLast = g.dBufXY();
      br.derivatives(bracketAParGLast_K, dBrLast);
      auto bracketTotalGLast_K = br.halfBracket(Grid::sliceXY(dGM, A_PAR), dBrLast);

      g.for_each_kxky([&](Dim kx, Dim ky) {
        GM_Nonlinear_K(kx, ky, G_MIN) = nonlinear::G2(
            bracketPhiG2_K(kx, ky), bracketAParG3_K(kx, ky), bracketAParUEKPar_K(kx, ky));
        GM_K_Star(kx, ky, G_MIN) = exp_nu(kx, ky) * moments_K(kx, ky, G_MIN) +
                                   dt / 2.0 * (1 + exp_nu(kx, ky)) * GM_Nonlinear_K(kx, ky, G_MIN);

        GM_Nonlinear_K(kx, ky, LAST) =
            nonlinear::GLast(bracketPhiGLast_K(kx, ky), bracketTotalGLast_K(kx, ky));
        GM_K_Star(kx, ky, LAST) =
            exp_gm(LAST) * exp_nu_g(kx, ky) * moments_K(kx, ky, LAST) +
            dt / 2.0 * (1 + exp_gm(LAST) * exp_nu_g(kx, ky)) * GM_Nonlinear_K(kx, ky, LAST);
      });

      auto dGMinusPlus = g.dBufXY();
      for (Dim m = G_MIN + 1; m < LAST; ++m) {
        g.for_each_xy([&](Dim x, Dim y) {
          dGMinusPlus.DX(x, y) =
              std::sqrt(m) * dGM.DX(x, y, m - 1) + std::sqrt(m + 1) * dGM.DX(x, y, m + 1);
          dGMinusPlus.DY(x, y) =
              std::sqrt(m) * dGM.DY(x, y, m - 1) + std::sqrt(m + 1) * dGM.DY(x, y, m + 1);
        });

        auto bracketAParGMMinusPlus_K = br.halfBracket(Grid::sliceXY(dGM, A_PAR), dGMinusPlus);
        auto bracketPhiGM_K = br.halfBracket(dPhi, Grid::sliceXY(dGM, m));

        g.for_each_kxky([&](Dim kx, Dim ky) {
          GM_Nonlinear_K(kx, ky, m) =
              nonlinear::GM(m, bracketPhiGM_K(kx, ky), bracketAParGMMinusPlus_K(kx, ky));
          GM_K_Star(kx, ky, m) =
              exp_gm(m) * exp_nu_g(kx, ky) * moments_K(kx, ky, m) +
              dt / 2.0 * (1 + exp_gm(m) * exp_nu_g(kx, ky)) * GM_Nonlinear_K(kx, ky, m);
        });
      }
    }

    // corrector step

    // Phi, Nabla, and other prep for A bracket
    g.for_each_kxky([&](Dim kx, Dim ky) {
      // set to 0 for (kx, ky)=(0,0)
      phi_K_New(kx, ky) =
          ((kx | ky) == 0) ? 0 : nonlinear::phi(GM_K_Star(kx, ky, N_E), g.kPerp2(kx, ky));
      ueKPar_K_New(kx, ky) = -g.kPerp2(kx, ky) * GM_K_Star(kx, ky, A_PAR);
    });

    auto dPhi_Loop = g.dBufXY(), dUEKPar_Loop = g.dBufXY();
    auto dGM_Loop = g.dBufMXY();
    br.derivatives(phi_K_New, dPhi_Loop);
    br.derivatives(ueKPar_K_New, dUEKPar_Loop);

    for (int m = 0; m < g.M; ++m) {
      // TODO(OPT) not necessary if we bail (only up to G_MIN)
      br.derivatives(Grid::sliceXY(GM_K_Star, m), Grid::sliceXY(dGM_Loop, m));
    }

    // Corrector loop
    // While momentsNew_K values don't directly depend on GM_K_Star,
    // they share a common subexpression so we use GM_K_Star to reduce recomputation.
    // TODO confirm that only m derivatives are needed at a time
    //  (if not, can always store one in a temporary buffer)

    auto guessAPar_K = g.cBufXY(), semiImplicitOperator = g.cBufXY();
    g.for_each_kxky([&](Dim kx, Dim ky) {
      guessAPar_K(kx, ky) = moments_K(kx, ky, A_PAR);
      semiImplicitOperator(kx, ky) = nonlinear::semiImplicitOp(dt, bPerpMax, aa0, g.kPerp2(kx, ky));
    });
    semiImplicitOperator(0, 0) = 0;

    Real old_error = 0, relative_error = 0;

    for (int p = 0; p <= MaxP; ++p) {
      auto DerivateNewMoment = [&](Dim m) {
        br.derivatives(Grid::sliceXY(momentsNew_K, m), Grid::sliceXY(dGM_Loop, m));
      };

      // First, compute A_par
      auto dPhiNeG2_Loop = g.dBufXY();
      g.for_each_xy([&](Dim x, Dim y) {
        if (g.M > 2) {
          dPhiNeG2_Loop.DX(x, y) =
              dPhi_Loop.DX(x, y) -
              rhoS * rhoS * (std::sqrt(2) * dGM_Loop.DX(x, y, G_MIN) + dGM_Loop.DX(x, y, N_E));
          dPhiNeG2_Loop.DY(x, y) =
              dPhi_Loop.DY(x, y) -
              rhoS * rhoS * (std::sqrt(2) * dGM_Loop.DY(x, y, G_MIN) + dGM_Loop.DY(x, y, N_E));
        } else {
          dPhiNeG2_Loop.DX(x, y) = dPhi_Loop.DX(x, y) - rhoS * rhoS * dGM_Loop.DX(x, y, N_E);
          dPhiNeG2_Loop.DY(x, y) = dPhi_Loop.DY(x, y) - rhoS * rhoS * dGM_Loop.DY(x, y, N_E);
        }
      });

      auto bracketAParPhiG2Ne_K_Loop =
          br.halfBracket(Grid::sliceXY(dGM_Loop, A_PAR), dPhiNeG2_Loop);
      auto bracketUEParPhi_K_Loop = br.halfBracket(dUEKPar_Loop, dPhi_Loop);

      /// f_pred from Viriato
      auto GM_Nonlinear_K_Loop = g.cBufMXY();
      Real sumAParRelError = 0;
      g.for_each_kxky([&](Dim kx, Dim ky) {
        GM_Nonlinear_K_Loop(kx, ky, A_PAR) = nonlinear::A(
            bracketAParPhiG2Ne_K_Loop(kx, ky), bracketUEParPhi_K_Loop(kx, ky), g.kPerp2(kx, ky));
        // TODO(OPT) reuse star
        momentsNew_K(kx, ky, A_PAR) =
            1.0 / (1.0 + semiImplicitOperator(kx, ky) / 4.0) *
            (GM_K_Star(kx, ky, A_PAR) +
             dt / 2.0 * (GM_Nonlinear_K_Loop(kx, ky, A_PAR) - GM_Nonlinear_K(kx, ky, A_PAR)) +
             semiImplicitOperator(kx, ky) / 4.0 * guessAPar_K(kx, ky));
        ueKPar_K_New(kx, ky) = -g.kPerp2(kx, ky) * momentsNew_K(kx, ky, A_PAR);

        sumAParRelError += std::norm(momentsNew_K(kx, ky, A_PAR) - moments_K(kx, ky, A_PAR));
      });

      old_error = relative_error;
      relative_error = 0;
      g.for_each_kxky([&](Dim kx, Dim ky) {
        relative_error =
            std::max(relative_error, std::abs(semiImplicitOperator(kx, ky) / 4.0 *
                                              (momentsNew_K(kx, ky, A_PAR) - guessAPar_K(kx, ky))) /
                                         std::sqrt(sumAParRelError / (Real(g.KX) * Real(g.KY))));
      });

      spdlog::debug("sumAParRelError: {}, relative_error: {}", sumAParRelError, relative_error);
      // TODO(OPT) bail if relative error is large

      DerivateNewMoment(A_PAR);
      br.derivatives(ueKPar_K_New, dUEKPar_Loop);

      auto bracketPhiNE_K_Loop = br.halfBracket(dPhi_Loop, Grid::sliceXY(dGM_Loop, N_E));
      auto bracketAParUEKPar_K_Loop = br.halfBracket(Grid::sliceXY(dGM_Loop, A_PAR), dUEKPar_Loop);

      g.for_each_kxky([&](Dim kx, Dim ky) {
        GM_Nonlinear_K_Loop(kx, ky, N_E) =
            nonlinear::N(bracketPhiNE_K_Loop(kx, ky), bracketAParUEKPar_K_Loop(kx, ky));
        momentsNew_K(kx, ky, N_E) =
            GM_K_Star(kx, ky, N_E) +
            dt / 2.0 * (GM_Nonlinear_K_Loop(kx, ky, N_E) - GM_Nonlinear_K(kx, ky, N_E));

        phi_K_New(kx, ky) =
            (kx | ky) == 0 ? 0 : nonlinear::phi(momentsNew_K(kx, ky, N_E), g.kPerp2(kx, ky));
      });

      br.derivatives(phi_K_New, dPhi_Loop);
      DerivateNewMoment(N_E);
      if (g.M > 2) {
        // Compute G2
        auto bracketPhiG2_K_Loop = br.halfBracket(dPhi_Loop, Grid::sliceXY(dGM_Loop, G_MIN));
        auto bracketAParG3_K_Loop =
            br.halfBracket(Grid::sliceXY(dGM_Loop, A_PAR), Grid::sliceXY(dGM_Loop, G_MIN + 1));

        g.for_each_kxky([&](Dim kx, Dim ky) {
          GM_Nonlinear_K_Loop(kx, ky, G_MIN) =
              nonlinear::G2(bracketPhiG2_K_Loop(kx, ky), bracketAParG3_K_Loop(kx, ky),
                            bracketAParUEKPar_K_Loop(kx, ky));
          momentsNew_K(kx, ky, G_MIN) =
              GM_K_Star(kx, ky, G_MIN) +
              dt / 2.0 * (GM_Nonlinear_K_Loop(kx, ky, G_MIN) - GM_Nonlinear_K(kx, ky, G_MIN));
        });
        DerivateNewMoment(G_MIN);

        DxDy<Buf::R_XY> dGMinusPlus_Loop = g.dBufXY();
        for (int m = G_MIN + 1; m < LAST; ++m) {
          g.for_each_xy([&](Dim x, Dim y) {
            dGMinusPlus_Loop.DX(x, y) = std::sqrt(m) * dGM_Loop.DX(x, y, m - 1) +
                                        std::sqrt(m + 1) * dGM_Loop.DX(x, y, m + 1);
            dGMinusPlus_Loop.DY(x, y) = std::sqrt(m) * dGM_Loop.DY(x, y, m - 1) +
                                        std::sqrt(m + 1) * dGM_Loop.DY(x, y, m + 1);
          });

          auto bracketAParGMMinusPlus_K_Loop =
              br.halfBracket(Grid::sliceXY(dGM_Loop, A_PAR), dGMinusPlus_Loop);
          auto bracketPhiGM_K_Loop = br.halfBracket(dPhi_Loop, Grid::sliceXY(dGM_Loop, m));

          g.for_each_kxky([&](Dim kx, Dim ky) {
            GM_Nonlinear_K_Loop(kx, ky, m) = nonlinear::GM(m, bracketPhiGM_K_Loop(kx, ky),
                                                           bracketAParGMMinusPlus_K_Loop(kx, ky));
            momentsNew_K(kx, ky, m) =
                GM_K_Star(kx, ky, m) +
                dt / 2.0 * (GM_Nonlinear_K_Loop(kx, ky, m) - GM_Nonlinear_K(kx, ky, m));
          });

          DerivateNewMoment(m);
        }

        // Compute G_{M-1}
        auto bracketPhiGLast_K_Loop = br.halfBracket(dPhi_Loop, Grid::sliceXY(dGM_Loop, LAST));
        auto bracketAParGLast_K_Loop =
            br.halfBracket(Grid::sliceXY(dGM_Loop, A_PAR), Grid::sliceXY(dGM_Loop, LAST));
        g.for_each_kxky([&](Dim kx, Dim ky) {
          bracketAParGLast_K_Loop(kx, ky) *=
              nonlinear::GLastBracketFactor(g.M, g.kPerp2(kx, ky), hyper);
          bracketAParGLast_K_Loop(kx, ky) +=
              rhoS / de * std::sqrt(LAST) * momentsNew_K(kx, ky, LAST - 1);
          // Note: Viriato adds this after derivative, but can be distributed
        });

        DxDy<Buf::R_XY> dBrLast_Loop = g.dBufXY();
        br.derivatives(bracketAParGLast_K_Loop, dBrLast_Loop);
        auto bracketTotalGLast_K_Loop =
            br.halfBracket(Grid::sliceXY(dGM_Loop, A_PAR), dBrLast_Loop);

        g.for_each_kxky([&](Dim kx, Dim ky) {
          GM_Nonlinear_K_Loop(kx, ky, LAST) =
              nonlinear::GLast(bracketPhiGLast_K_Loop(kx, ky), bracketTotalGLast_K_Loop(kx, ky));
          // TODO(OPT) reuse star
          momentsNew_K(kx, ky, LAST) =
              GM_K_Star(kx, ky, LAST) +
              dt / 2.0 * (GM_Nonlinear_K_Loop(kx, ky, LAST) - GM_Nonlinear_K(kx, ky, LAST));
        });
        DerivateNewMoment(LAST);
      }
      if (relative_error <= epsilon) {
        spdlog::debug("Converged at p={} with relative_error={}", p, relative_error);
        break;
      }
      if (p != 0 and relative_error / old_error > 1.0) {
        spdlog::debug("Diverging at p={} with relative_error={} & old_error={}", p, relative_error,
                      old_error);
        divergent = true;
        divergentCount++;
        dt = low * dt;
        break;
      }
      if (relative_error > epsilon and p == MaxP) {
        // did not converge well enough
        spdlog::debug("Repeating!");
        repeat = true;
        repeatCount++;
        dt = low * dt;
        break;
      }

      g.for_each_kxky([&](Dim kx, Dim ky) { guessAPar_K(kx, ky) = momentsNew_K(kx, ky, A_PAR); });
    }

    if (divergent) { continue; }
    if (repeat) {
      noInc = true;
      continue;
    }

    // Next timestep
    t++;
    this->elapsedT += dt;

    // Update dt
    Real tempDt =
        getTimestep(dPhi_Loop, Grid::sliceXY(dGM_Loop, N_E), Grid::sliceXY(dGM_Loop, A_PAR));
    dt = updateTimestep(dt, tempDt, noInc, relative_error);

    spdlog::info("Moving on to next timestep: {}\n"
                 "dt is: {}",
                 t, dt);
    noInc = false;
    saved = false;

    // New values are now old. Old values will be overwritten in the next timestep.
    std::swap(moments_K, momentsNew_K);
    std::swap(phi_K, phi_K_New);
    std::swap(ueKPar_K, ueKPar_K_New);

    // Also swap derivatives
    std::swap(dPhi, dPhi_Loop);
    std::swap(dUEKPar, dUEKPar_Loop);
    std::swap(dGM, dGM_Loop);

    // Must be after swap for now, it looks at current, not new values
    auto [magnetic, kinetic] = calculateEnergies();
    spdlog::info("t={} magnetic energy: {}, kinetic energy: {}", t, magnetic, kinetic);

    // Log moment values when level is trace (most verbose)
    for (Dim m = 0; m < g.M; ++m) {
      spdlog::trace(
          "t={} m={}:\n{}", t, m,
          fmt::streamed(ostream_tuple(std::setprecision(16), Grid::sliceXY(moments_K, m))));
    }
  }

  spdlog::info("Finished! Repeat count: {}, Divergent count: {}", repeatCount, divergentCount);

  // TODO need a way to only export the final timestep
  if (saveInterval != 0) { exportTimestep(N); }
}

void Naive::exportTimestep(Dim t) {
  std::ostringstream oss;
  oss << "a_par_t" << t << ".npy";
  exporter.exportTo(oss.str(), Grid::sliceXY(moments_K, A_PAR));

  oss.str("");
  oss << "phi_t" << t << ".npy";
  exporter.exportTo(oss.str(), phi_K);

  oss.str("");
  oss << "uekpar_t" << t << ".npy";
  exporter.exportTo(oss.str(), ueKPar_K);
}

Real Naive::updateTimestep(Real dt, Real tempDt, bool noInc, Real relative_error) const {
  Real inc_factor = noInc ? 1 : 1.08;

  if (relative_error < 0.8 * epsilon) { dt *= inc_factor; }
  dt = std::min(tempDt, dt);
  return dt;
}

Naive::Buf::R_XY Naive::getMoment(Dim m) const {
  // Make a copy first
  Buf::C_XY tmp = g.cBufXY();
  g.for_each_kxky([&](Dim kx, Dim ky) { tmp(kx, ky) = moments_K(kx, ky, m); });

  Buf::R_XY out = g.rBufXY();
  tf.bfft(tmp, out);

  return out;
}

Real Naive::getTimestep(DxDy<View::R_XY> dPhi, DxDy<View::R_XY> dNE, DxDy<View::R_XY> dAPar) {
  // compute flows
  DxDy<View::R_XY> ve, b;
  Real vyMax{0}, vxMax{0}, bxMax{0}, byMax{0};
  bPerpMax = 0;

  // Note that this is minus in Viriato, but we don't care because we're taking the absolute value
  // anyway.
  ve.DX = dPhi.DY;
  ve.DY = dPhi.DX;
  b.DX = dAPar.DY;
  b.DY = dAPar.DX;

  g.for_each_xy([&](Dim x, Dim y) {
    bxMax = std::max(bxMax, std::abs(b.DX(x, y)));
    byMax = std::max(byMax, std::abs(b.DY(x, y)));
    bPerpMax = std::max(bPerpMax, std::sqrt(b.DX(x, y) * b.DX(x, y) + b.DY(x, y) * b.DY(x, y)));
    vxMax = std::max(vxMax, std::abs(ve.DX(x, y)));
    vyMax = std::max(vyMax, std::abs(ve.DY(x, y)));
    if (rhoI >= smallRhoI) {
      vxMax = std::max(vxMax, rhoS * rhoS * std::abs(dNE.DX(x, y)));
      vyMax = std::max(vyMax, rhoS * rhoS * std::abs(dNE.DY(x, y)));
    }
  });

  Real kperpDum2 = std::pow(g.ky_(g.KY / 2), 2) + std::pow(Real(g.KX), 2);
  Real omegaKaw;
  if (rhoI < smallRhoI) {
    omegaKaw = std::sqrt(1.0 + kperpDum2 * (3.0 / 4.0 * rhoI * rhoI + rhoS * rhoS)) *
               g.ky_(g.KY / 2) * bPerpMax / (1.0 + kperpDum2 * de * de);
  } else {
    omegaKaw =
        std::sqrt(kperpDum2 *
                  (rhoS * rhoS - rhoI * rhoI / (Gamma0(0.5 * kperpDum2 * rhoI * rhoI) - 1.0))) *
        g.ky_(g.KY / 2 + 1) * bPerpMax / std::sqrt(1.0 + kperpDum2 * de * de);
  }

  Real dx = lx / Real(g.X), dy = ly / Real(g.Y);

  Real CFLFlow;
  if (g.M > 2) {
    CFLFlow = std::min({dx / vxMax, dy / vyMax, 2.0 / omegaKaw,
                        std::min(dx / bxMax, dy / byMax) / (rhoS / de) / std::sqrt(LAST)});
  } else {
    CFLFlow = std::min({dx / vxMax, dy / vyMax, 2.0 / omegaKaw, dx / bxMax, dy / byMax});
  }

  spdlog::debug("vxmax: {}, vymax: {}, bxmax: {}, bymax: {}", vxMax, vyMax, bxMax, byMax);
  spdlog::debug("bperp_max: {}, omegakaw: {}, CFLFlow: {}", bPerpMax, omegaKaw, CFLFlow);
  spdlog::debug("Calculated timestep: {}", CFLFrac * CFLFlow);

  return CFLFrac * CFLFlow;
}

Naive::Energies Naive::calculateEnergies() const {
  Energies e{};
  g.for_each_kxky([&](Dim kx, Dim ky) {
    Real const factor = kx == 0 ? 0.5 : 1.0;
    e.magnetic += factor * g.kPerp2(kx, ky) * std::norm(moments_K(kx, ky, A_PAR));
    if (rhoI < smallRhoI) {
      e.kinetic += factor * g.kPerp2(kx, ky) * std::norm(phi_K(kx, ky));
    } else {
      e.kinetic -= factor * 1.0 / (rhoI * rhoI) *
                   (Gamma0(g.kPerp2(kx, ky) * rhoI * rhoI / 2.0) - 1) * std::norm(phi_K(kx, ky));
    }
  });

  // Normalize the energies
  e.magnetic *= XYNorm;
  e.kinetic *= XYNorm;

  return e;
}
} // namespace ahr
