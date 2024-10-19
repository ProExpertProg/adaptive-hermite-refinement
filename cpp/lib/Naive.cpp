#include "Naive.hpp"
#include "equillibrium.hpp"

#include <cnpy.h>
#include <cstdlib>
#include <utility>

namespace {
template <class T, class... Args, size_t... I>
auto forward_to_array_impl(std::index_sequence<I...> indices, Args &&...args) {
  return std::array<T, indices.size()>{
      [&args...](size_t index) { return T{std::forward<Args>(args)...}; }(I)...};
}

template <class T, size_t Size, class... Args>
std::array<T, Size> forward_to_array(Args &&...args) {
  return forward_to_array_impl<T>(std::make_index_sequence<Size>(), std::forward<Args>(args)...);
}
} // namespace

namespace ahr {
Naive::Naive(std::ostream &out, Dim M, Dim X, Dim Y) : HermiteRunner(out), M(M), X(X), Y(Y) {
  assert(M >= 4 or M == 2);
  // X and Y must be powers of 2
  assert((X & (X - 1)) == 0);
  assert((Y & (Y - 1)) == 0);
}

void Naive::hlFilter(CViewXY &complexArray) {
  for_each_kxky([&](Dim kx, Dim ky) {
    complexArray(kx, ky) *=
        exp(-36.0 * pow(kx_(kx) / KX, 36.0)) * exp(-36.0 * pow(ky_(ky) / KY, 36.0));
  });
}

void Naive::fft(ViewXY in, CViewXY out) {
  fft_base(in, out);
  hlFilter(out);
}

void Naive::init(std::string_view equilibriumName) {
  // Currently assuming X==Y for simplicity, but the code is written generally for the most part.
  assert(X == Y);
  Buf2D temp{X, Y};

  // Plan FFTs both ways
  fft_base = fftw::plan_r2c<2u>::dft(temp.to_mdspan(), phi_K.to_mdspan(), fftw::ESTIMATE);
  fftInv = fftw::plan_c2r<2u>::dft(phi_K.to_mdspan(), temp.to_mdspan(), fftw::ESTIMATE);

  // Initialize equilibrium values
  auto [aParEq, phi] = equilibrium(equilibriumName, X, Y);

  fft(phi.to_mdspan(), phi_K.to_mdspan());
  fft(aParEq.to_mdspan(), aParEq_K.to_mdspan());

  // Transform moments into phase space
  for (int m = G_MIN; m < M; ++m) {
    for_each_kxky([&](Dim kx, Dim ky) { moments_K(kx, ky, m) = 0; });
  }

  // aPar, uekPar, ne
  for_each_kxky([&](Dim kx, Dim ky) {
    moments_K(kx, ky, N_E) = nonlinear::phiInv(phi_K(kx, ky), kPerp2(kx, ky));
    moments_K(kx, ky, A_PAR) = aParEq_K(kx, ky);
    ueKPar_K(kx, ky) = -kPerp2(kx, ky) * moments_K(kx, ky, A_PAR);
  });
}

void Naive::run(Dim N, Dim saveInterval) {
  // store all derivatives
  DxDy<Buf3D> dGM{X, Y, M};
  DxDy<Buf2D> dPhi{X, Y}, dUEKPar{X, Y};

  // isothermal if only running with 2 moments
  assert(M >= 4 or M == 2);

  bool divergent = false, repeat = false, noInc = false;
  int divergentCount = 0, repeatCount = 0;
  HyperCoefficients hyper{};

  bool saved = false;
  // Manually increment t only if not diverging
  for (Dim t = 0; t < N;) {
    if (saveInterval != 0 and t % saveInterval == 0) {
      if (!saved) {
        out << "Saving for timestep: " << t << std::endl;
        saved = true;
        exportTimestep(t);
      }
    }

    // predictor step
    derivatives(phi_K, dPhi);
    derivatives(ueKPar_K, dUEKPar);
    for (int m = 0; m < M; ++m) {
      derivatives(sliceXY(moments_K, m), sliceXY(dGM, m));
    }

    if (repeat or divergent) {
      out << std::boolalpha << "repeat: " << repeat << ", divergent:" << divergent << std::endl;
      repeat = false;
      divergent = false;
    } else if (dt == -1) {
      dt = getTimestep(dPhi, sliceXY(dGM, N_E), sliceXY(dGM, A_PAR));
      hyper = HyperCoefficients::calculate(dt, KX, KY, M);
    }

    // DEBUG
    out << "dt: " << dt << std::endl;

    // store results of nonlinear operators, as well as results of predictor step
    Buf3D_K GM_K_Star{KX, KY, M}, GM_Nonlinear_K{KX, KY, M};

    // Compute N
    auto bracketPhiNE_K = halfBracket(dPhi, sliceXY(dGM, N_E));
    auto bracketAParUEKPar_K = halfBracket(sliceXY(dGM, A_PAR), dUEKPar);

    // Compute A
    DxDy<Buf2D> dPhiNeG2{X, Y};
    if (M > 2) {
      for_each_xy([&](Dim x, Dim y) {
        dPhiNeG2.DX(x, y) =
            dPhi.DX(x, y) - rhoS * rhoS * (std::sqrt(2) * dGM.DX(x, y, G_MIN) + dGM.DX(x, y, N_E));
        dPhiNeG2.DY(x, y) =
            dPhi.DY(x, y) - rhoS * rhoS * (std::sqrt(2) * dGM.DY(x, y, G_MIN) + dGM.DY(x, y, N_E));
      });
    } else {
      for_each_xy([&](Dim x, Dim y) {
        dPhiNeG2.DX(x, y) = dPhi.DX(x, y) - rhoS * rhoS * dGM.DX(x, y, N_E);
        dPhiNeG2.DY(x, y) = dPhi.DY(x, y) - rhoS * rhoS * dGM.DY(x, y, N_E);
      });
    }

    auto bracketAParPhiG2Ne_K = halfBracket(sliceXY(dGM, A_PAR), dPhiNeG2);
    auto bracketUEParPhi_K = halfBracket(dUEKPar, dPhi);

    for_each_kxky([&](Dim kx, Dim ky) {
      GM_Nonlinear_K(kx, ky, N_E) =
          nonlinear::N(bracketPhiNE_K(kx, ky), bracketAParUEKPar_K(kx, ky));
      GM_K_Star(kx, ky, N_E) =
          exp_nu(kx, ky, hyper.nu_2, dt) * moments_K(kx, ky, N_E) +
          dt / 2.0 * (1 + exp_nu(kx, ky, hyper.nu_2, dt)) * GM_Nonlinear_K(kx, ky, N_E);

      GM_Nonlinear_K(kx, ky, A_PAR) =
          nonlinear::A(bracketAParPhiG2Ne_K(kx, ky), bracketUEParPhi_K(kx, ky), kPerp2(kx, ky));
      GM_K_Star(kx, ky, A_PAR) =
          exp_eta(kx, ky, hyper.eta2, dt) * moments_K(kx, ky, A_PAR) +
          dt / 2.0 * (1 + exp_eta(kx, ky, hyper.eta2, dt)) * GM_Nonlinear_K(kx, ky, A_PAR) +
          (1.0 - exp_eta(kx, ky, hyper.eta2, dt)) * aParEq_K(kx, ky);
    });

    if (M > 2) {
      // Compute G2
      auto bracketPhiG2_K = halfBracket(dPhi, sliceXY(dGM, G_MIN));
      auto bracketAParG3_K = halfBracket(sliceXY(dGM, A_PAR), sliceXY(dGM, G_MIN + 1));

      // Compute G_{M-1}
      auto bracketPhiGLast_K = halfBracket(dPhi, sliceXY(dGM, LAST));
      auto bracketAParGLast_K = halfBracket(sliceXY(dGM, A_PAR), sliceXY(dGM, LAST));
      for_each_kxky([&](Dim kx, Dim ky) {
        bracketAParGLast_K(kx, ky) *= nonlinear::GLastBracketFactor(M, kPerp2(kx, ky), hyper);
        bracketAParGLast_K(kx, ky) += rhoS / de * std::sqrt(LAST) * moments_K(kx, ky, LAST - 1);
        // TODO Viriato adds this after the derivative
      });

      DxDy<Buf2D> dBrLast{X, Y};
      derivatives(bracketAParGLast_K, dBrLast);
      auto bracketTotalGLast_K = halfBracket(sliceXY(dGM, A_PAR), dBrLast);

      for_each_kxky([&](Dim kx, Dim ky) {
        GM_Nonlinear_K(kx, ky, G_MIN) = nonlinear::G2(
            bracketPhiG2_K(kx, ky), bracketAParG3_K(kx, ky), bracketAParUEKPar_K(kx, ky));
        GM_K_Star(kx, ky, G_MIN) =
            exp_nu(kx, ky, hyper.nu_2, dt) * moments_K(kx, ky, G_MIN) +
            dt / 2.0 * (1 + exp_nu(kx, ky, hyper.nu_2, dt)) * GM_Nonlinear_K(kx, ky, G_MIN);

        GM_Nonlinear_K(kx, ky, LAST) =
            nonlinear::GLast(bracketPhiGLast_K(kx, ky), bracketTotalGLast_K(kx, ky));
        GM_K_Star(kx, ky, LAST) =
            exp_gm(LAST, hyper.nu_ei, dt) * exp_nu(kx, ky, hyper.nu_g, dt) *
                moments_K(kx, ky, LAST) +
            dt / 2.0 * (1 + exp_gm(LAST, hyper.nu_ei, dt) * exp_nu(kx, ky, hyper.nu_g, dt)) *
                GM_Nonlinear_K(kx, ky, LAST);
      });

      DxDy<Buf2D> dGMinusPlus{X, Y};
      for (Dim m = G_MIN + 1; m < LAST; ++m) {
        for_each_xy([&](Dim x, Dim y) {
          dGMinusPlus.DX(x, y) =
              std::sqrt(m) * dGM.DX(x, y, m - 1) + std::sqrt(m + 1) * dGM.DX(x, y, m + 1);
          dGMinusPlus.DY(x, y) =
              std::sqrt(m) * dGM.DY(x, y, m - 1) + std::sqrt(m + 1) * dGM.DY(x, y, m + 1);
        });

        auto bracketAParGMMinusPlus_K = halfBracket(sliceXY(dGM, A_PAR), dGMinusPlus);
        auto bracketPhiGM_K = halfBracket(dPhi, sliceXY(dGM, m));

        for_each_kxky([&](Dim kx, Dim ky) {
          GM_Nonlinear_K(kx, ky, m) =
              nonlinear::GM(m, bracketPhiGM_K(kx, ky), bracketAParGMMinusPlus_K(kx, ky));
          GM_K_Star(kx, ky, m) =
              exp_gm(m, hyper.nu_ei, dt) * exp_nu(kx, ky, hyper.nu_g, dt) * moments_K(kx, ky, m) +
              dt / 2.0 * (1 + exp_gm(m, hyper.nu_ei, dt) * exp_nu(kx, ky, hyper.nu_g, dt)) *
                  GM_Nonlinear_K(kx, ky, m);
        });
      }
    }

    // corrector step

    // Phi, Nabla, and other prep for A bracket
    for_each_kxky([&](Dim kx, Dim ky) {
      // set to 0 for (kx, ky)=(0,0)
      phi_K_New(kx, ky) =
          ((kx | ky) == 0) ? 0 : nonlinear::phi(GM_K_Star(kx, ky, N_E), kPerp2(kx, ky));
      ueKPar_K_New(kx, ky) = -kPerp2(kx, ky) * GM_K_Star(kx, ky, A_PAR);
    });

    DxDy<Buf2D> dPhi_Loop{X, Y}, dUEKPar_Loop{X, Y};
    DxDy<Buf3D> dGM_Loop{X, Y, M};
    derivatives(phi_K_New, dPhi_Loop);
    derivatives(ueKPar_K_New, dUEKPar_Loop);

    for (int m = 0; m < M; ++m) {
      // TODO(OPT) not necessary if we bail (only up to G_MIN)
      derivatives(sliceXY(GM_K_Star, m), sliceXY(dGM_Loop, m));
    }

    // Corrector loop
    // TODO confirm that only m derivatives are needed at a time
    //  (if not, can always store one in a temporary buffer)

    Buf2D_K guessAPar_K{KX, KY}, semiImplicitOperator{KX, KY};
    for_each_kxky([&](Dim kx, Dim ky) {
      guessAPar_K(kx, ky) = moments_K(kx, ky, A_PAR);
      semiImplicitOperator(kx, ky) = nonlinear::semiImplicitOp(dt, bPerpMax, aa0, kPerp2(kx, ky));
    });
    semiImplicitOperator(0, 0) = 0;

    Real old_error = 0, relative_error = 0;

    for (int p = 0; p <= MaxP; ++p) {
      auto DerivateNewMoment = [&](Dim m) {
        derivatives(sliceXY(momentsNew_K, m), sliceXY(dGM_Loop, m));
      };

      // First, compute A_par
      DxDy<Buf2D> dPhiNeG2_Loop{X, Y};
      for_each_xy([&](Dim x, Dim y) {
        if (M > 2) {
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

      auto bracketAParPhiG2Ne_K_Loop = halfBracket(sliceXY(dGM_Loop, A_PAR), dPhiNeG2_Loop);
      auto bracketUEParPhi_K_Loop = halfBracket(dUEKPar_Loop, dPhi_Loop);

      /// f_pred from Viriato
      Buf3D_K GM_Nonlinear_K_Loop{KX, KY, M};
      Real sumAParRelError = 0;
      for_each_kxky([&](Dim kx, Dim ky) {
        GM_Nonlinear_K_Loop(kx, ky, A_PAR) = nonlinear::A(
            bracketAParPhiG2Ne_K_Loop(kx, ky), bracketUEParPhi_K_Loop(kx, ky), kPerp2(kx, ky));
        // TODO(OPT) reuse star
        momentsNew_K(kx, ky, A_PAR) =
            1.0 / (1.0 + semiImplicitOperator(kx, ky) / 4.0) *
            (exp_eta(kx, ky, hyper.eta2, dt) * moments_K(kx, ky, A_PAR) +
             dt / 2.0 * exp_eta(kx, ky, hyper.eta2, dt) * GM_Nonlinear_K(kx, ky, A_PAR) +
             dt / 2.0 * GM_Nonlinear_K_Loop(kx, ky, A_PAR) +
             (1.0 - exp_eta(kx, ky, hyper.eta2, dt)) * aParEq_K(kx, ky) +
             semiImplicitOperator(kx, ky) / 4.0 * guessAPar_K(kx, ky));
        ueKPar_K_New(kx, ky) = -kPerp2(kx, ky) * momentsNew_K(kx, ky, A_PAR);

        sumAParRelError += std::norm(momentsNew_K(kx, ky, A_PAR) - moments_K(kx, ky, A_PAR));
      });

      old_error = relative_error;
      relative_error = 0;
      for_each_kxky([&](Dim kx, Dim ky) {
        relative_error =
            std::max(relative_error, std::abs(semiImplicitOperator(kx, ky) / 4.0 *
                                              (momentsNew_K(kx, ky, A_PAR) - guessAPar_K(kx, ky))) /
                                         std::sqrt(sumAParRelError / (Real(KX) * Real(KY))));
      });

      out << "sumApar relative_error:" << sumAParRelError << std::endl;
      out << "relative_error:" << relative_error << std::endl;
      // TODO(OPT) bail if relative error is large

      DerivateNewMoment(A_PAR);
      derivatives(ueKPar_K_New, dUEKPar_Loop);

      auto bracketPhiNE_K_Loop = halfBracket(dPhi_Loop, sliceXY(dGM_Loop, N_E));
      auto bracketAParUEKPar_K_Loop = halfBracket(sliceXY(dGM_Loop, A_PAR), dUEKPar_Loop);

      for_each_kxky([&](Dim kx, Dim ky) {
        GM_Nonlinear_K_Loop(kx, ky, N_E) =
            nonlinear::N(bracketPhiNE_K_Loop(kx, ky), bracketAParUEKPar_K_Loop(kx, ky));
        // TODO(OPT) reuse star
        momentsNew_K(kx, ky, N_E) =
            exp_nu(kx, ky, hyper.nu_2, dt) * moments_K(kx, ky, N_E) +
            dt / 2.0 * exp_nu(kx, ky, hyper.nu_2, dt) * GM_Nonlinear_K(kx, ky, N_E) +
            dt / 2.0 * GM_Nonlinear_K_Loop(kx, ky, N_E);

        phi_K_New(kx, ky) =
            (kx | ky) == 0 ? 0 : nonlinear::phi(momentsNew_K(kx, ky, N_E), kPerp2(kx, ky));
      });

      derivatives(phi_K_New, dPhi_Loop);
      DerivateNewMoment(N_E);
      if (M > 2) {
        // Compute G2
        auto bracketPhiG2_K_Loop = halfBracket(dPhi_Loop, sliceXY(dGM_Loop, G_MIN));
        auto bracketAParG3_K_Loop =
            halfBracket(sliceXY(dGM_Loop, A_PAR), sliceXY(dGM_Loop, G_MIN + 1));

        for_each_kxky([&](Dim kx, Dim ky) {
          GM_Nonlinear_K_Loop(kx, ky, G_MIN) =
              nonlinear::G2(bracketPhiG2_K_Loop(kx, ky), bracketAParG3_K_Loop(kx, ky),
                            bracketAParUEKPar_K_Loop(kx, ky));
          // TODO(OPT) reuse star
          momentsNew_K(kx, ky, G_MIN) =
              exp_nu(kx, ky, hyper.nu_2, dt) * moments_K(kx, ky, G_MIN) +
              dt / 2.0 * exp_nu(kx, ky, hyper.nu_2, dt) * GM_Nonlinear_K(kx, ky, G_MIN) +
              dt / 2.0 * GM_Nonlinear_K_Loop(kx, ky, G_MIN);
        });
        DerivateNewMoment(G_MIN);

        DxDy<Buf2D> dGMinusPlus_Loop{X, Y};
        for (int m = G_MIN + 1; m < LAST; ++m) {
          for_each_xy([&](Dim x, Dim y) {
            dGMinusPlus_Loop.DX(x, y) = std::sqrt(m) * dGM_Loop.DX(x, y, m - 1) +
                                        std::sqrt(m + 1) * dGM_Loop.DX(x, y, m + 1);
            dGMinusPlus_Loop.DY(x, y) = std::sqrt(m) * dGM_Loop.DY(x, y, m - 1) +
                                        std::sqrt(m + 1) * dGM_Loop.DY(x, y, m + 1);
          });

          auto bracketAParGMMinusPlus_K_Loop =
              halfBracket(sliceXY(dGM_Loop, A_PAR), dGMinusPlus_Loop);
          auto bracketPhiGM_K_Loop = halfBracket(dPhi_Loop, sliceXY(dGM_Loop, m));

          for_each_kxky([&](Dim kx, Dim ky) {
            GM_Nonlinear_K_Loop(kx, ky, m) = nonlinear::GM(m, bracketPhiGM_K_Loop(kx, ky),
                                                           bracketAParGMMinusPlus_K_Loop(kx, ky));
            // TODO(OPT) reuse star
            momentsNew_K(kx, ky, m) =
                exp_gm(m, hyper.nu_ei, dt) * exp_nu(kx, ky, hyper.nu_g, dt) * moments_K(kx, ky, m) +
                dt / 2.0 * exp_gm(m, hyper.nu_ei, dt) * exp_nu(kx, ky, hyper.nu_g, dt) *
                    GM_Nonlinear_K(kx, ky, m) +
                dt / 2.0 * GM_Nonlinear_K_Loop(kx, ky, m);
          });

          DerivateNewMoment(m);
        }

        // Compute G_{M-1}
        auto bracketPhiGLast_K_Loop = halfBracket(dPhi_Loop, sliceXY(dGM_Loop, LAST));
        auto bracketAParGLast_K_Loop =
            halfBracket(sliceXY(dGM_Loop, A_PAR), sliceXY(dGM_Loop, LAST));
        for_each_kxky([&](Dim kx, Dim ky) {
          bracketAParGLast_K_Loop(kx, ky) *=
              nonlinear::GLastBracketFactor(M, kPerp2(kx, ky), hyper);
          bracketAParGLast_K_Loop(kx, ky) +=
              rhoS / de * std::sqrt(LAST) * momentsNew_K(kx, ky, LAST - 1);
          // Note: Viriato adds this after derivative, but can be distributed
        });

        DxDy<Buf2D> dBrLast_Loop{X, Y};
        derivatives(bracketAParGLast_K_Loop, dBrLast_Loop);
        auto bracketTotalGLast_K_Loop = halfBracket(sliceXY(dGM_Loop, A_PAR), dBrLast_Loop);

        for_each_kxky([&](Dim kx, Dim ky) {
          GM_Nonlinear_K_Loop(kx, ky, LAST) =
              nonlinear::GLast(bracketPhiGLast_K_Loop(kx, ky), bracketTotalGLast_K_Loop(kx, ky));
          // TODO(OPT) reuse star
          momentsNew_K(kx, ky, LAST) =
              exp_gm(LAST, hyper.nu_ei, dt) * exp_nu(kx, ky, hyper.nu_g, dt) *
                  moments_K(kx, ky, LAST) +
              dt / 2.0 * exp_gm(LAST, hyper.nu_ei, dt) * exp_nu(kx, ky, hyper.nu_g, dt) *
                  GM_Nonlinear_K(kx, ky, LAST) +
              dt / 2.0 * GM_Nonlinear_K_Loop(kx, ky, LAST);
        });
        DerivateNewMoment(LAST);
      }
      if (relative_error <= epsilon) {
        out << "relative error small:" << relative_error << std::endl;
        break;
      }
      if (p != 0 and relative_error / old_error > 1.0) {
        out << "diverging!" << std::endl;
        divergent = true;
        divergentCount++;
        dt = low * dt;
        break;
      }
      if (relative_error > epsilon and p == MaxP) {
        // did not converge well enough
        out << "repeating!" << std::endl;
        repeat = true;
        repeatCount++;
        dt = low * dt;
        break;
      }

      for_each_kxky([&](Dim kx, Dim ky) { guessAPar_K(kx, ky) = momentsNew_K(kx, ky, A_PAR); });
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
    Real tempDt = getTimestep(dPhi_Loop, sliceXY(dGM_Loop, N_E), sliceXY(dGM_Loop, A_PAR));
    dt = updateTimestep(dt, tempDt, noInc, relative_error);
    hyper = HyperCoefficients::calculate(dt, KX, KY, M);

    out << "Moving on to next timestep: " << t << std::endl;
    out << "dti is: " << dt << std::endl;
    noInc = false;
    saved = false;

    // New values are now old. Old values will be overwritten in the next timestep.
    std::swap(moments_K, momentsNew_K);
    std::swap(phi_K, phi_K_New);
    std::swap(ueKPar_K, ueKPar_K_New);

    // Must be after swap for now, it looks at current, not new values
    auto [magnetic, kinetic] = calculateEnergies();
    out << "magnetic energy: " << magnetic << ", kinetic energy: " << kinetic << std::endl;
  }

  out << "dt actual: " << dt << std::endl;
  out << "repeat count: " << repeatCount << std::endl
      << "divergent count: " << divergentCount << std::endl;

  // TODO need a way to only export the final timestep
  if (saveInterval != 0) { exportTimestep(N); }
}

void Naive::exportTimestep(Dim t) {
  std::ostringstream oss;
  oss << "a_par_t" << t << ".npy";
  exportToNpy(oss.str(), sliceXY(moments_K, A_PAR));

  oss.str("");
  oss << "phi_t" << t << ".npy";
  exportToNpy(oss.str(), phi_K);

  oss.str("");
  oss << "uekpar_t" << t << ".npy";
  exportToNpy(oss.str(), ueKPar_K);
}

Real Naive::updateTimestep(Real dt, Real tempDt, bool noInc, Real relative_error) const {
  Real inc_factor = noInc ? 1 : 1.08;

  if (relative_error < 0.8 * epsilon) { dt *= inc_factor; }
  dt = std::min(tempDt, dt);
  return dt;
}

mdarray<Real, dextents<Dim, 2u>> Naive::getFinalAPar() {
  Buf2D buf{X, Y};
  // This actually wrecks A_PAR, but we don't need it anymore
  fftInv(sliceXY(moments_K, A_PAR), buf.to_mdspan());

  // Write to a layout_right array and normalize
  mdarray<Real, dextents<Dim, 2u>> result{X, Y};
  for_each_xy([&](Dim x, Dim y) { result(x, y) = buf(x, y) / double(X) / double(Y); });

  return result;
}

Naive::Buf2D Naive::getMoment(Dim m) const {
  // Make a copy first
  Buf2D_K tmp{KX, KY};
  for_each_kxky([&](Dim kx, Dim ky) { tmp(kx, ky) = moments_K(kx, ky, m); });

  Buf2D out{X, Y};
  fftInv(tmp.to_mdspan(), out.to_mdspan());

  return out;
}

[[nodiscard]] Naive::Buf2D_K Naive::fullBracket(Naive::CViewXY op1, Naive::CViewXY op2) {
  DxDy<Buf2D> derOp1{X, Y}, derOp2{X, Y};
  derivatives(op1, derOp1);
  derivatives(op2, derOp2);

  return halfBracket(derOp1, derOp2);
}

void Naive::derivatives(const Naive::CViewXY &op, Naive::DxDy<Naive::ViewXY> output) {
  DxDy<Buf2D_K> Der_K{KX, KY};
  prepareDXY_PH(op, Der_K.DX, Der_K.DY);
  fftInv(Der_K.DX.to_mdspan(), output.DX);
  fftInv(Der_K.DY.to_mdspan(), output.DY);
}

Naive::Buf2D_K Naive::halfBracket(Naive::DxDy<Naive::ViewXY> derOp1,
                                  Naive::DxDy<Naive::ViewXY> derOp2) {
  Buf2D br{X, Y};
  Buf2D_K br_K{KX, KY};
  bracket(derOp1, derOp2, br);
  fft(br.to_mdspan(), br_K.to_mdspan());
  br_K(0, 0) = 0;
  return br_K;
}

void Naive::exportToNpy(std::string path, ahr::Naive::ViewXY view) const {
  // Coordinates are flipped because we use layout_left
  cnpy::npy_save(std::move(path), view.data_handle(), {Y, X}, "w");
}

void Naive::exportToNpy(std::string path, ahr::Naive::CViewXY view) const {
  // fft overwrites the input, so we need to copy it to a temporary buffer
  Buf2D_K tempK{KX, KY};
  Buf2D temp{X, Y};

  for_each_kxky([&](Dim kx, Dim ky) { tempK(kx, ky) = view(kx, ky); });

  fftInv(tempK.to_mdspan(), temp.to_mdspan());
  normalize(temp.to_mdspan(), temp.to_mdspan());

  exportToNpy(std::move(path), temp.to_mdspan());
}

void Naive::normalize(Naive::ViewXY view, Naive::ViewXY viewOut) const {
  for_each_xy([&](Dim x, Dim y) { viewOut(x, y) = view(x, y) / double(X) / double(Y); });
}

Naive::Energies Naive::calculateEnergies() const {
  Energies e{};
  for_each_kxky([&](Dim kx, Dim ky) {
    Real const factor = kx == 0 ? 0.5 : 1.0;
    e.magnetic += factor * kPerp2(kx, ky) * std::norm(moments_K(kx, ky, A_PAR));
    if (rhoI < smallRhoI) {
      e.kinetic += factor * kPerp2(kx, ky) * std::norm(phi_K(kx, ky));
    } else {
      e.kinetic -= factor * 1.0 / (rhoI * rhoI) * (Gamma0(kPerp2(kx, ky) * rhoI * rhoI / 2.0) - 1) *
                   std::norm(phi_K(kx, ky));
    }
  });

  // Normalize the energies
  e.magnetic /= Real(X) * Real(Y);
  e.kinetic /= Real(X) * Real(Y);

  return e;
}
} // namespace ahr
