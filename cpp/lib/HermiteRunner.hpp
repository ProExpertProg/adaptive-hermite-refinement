#pragma once

#include "typedefs.hpp"
#include <experimental/mdarray>
#include <experimental/mdspan>
#include <iostream>

namespace ahr {
using std::experimental::dextents;
using std::experimental::mdarray;
using std::experimental::mdspan;

class HermiteRunner {
public:
  virtual ~HermiteRunner() = default;

  /**
   * init prepares the hermite simulation.
   */
  virtual void init(std::string_view equilibriumName) = 0;

  /**
   * run() will simulate Hermite moments for N timesteps.
   * init() must have been called before this call.
   * @param N the number of timesteps.
   */
  virtual void run(Dim N, Dim saveInterval) = 0;

  /**
   *
   * @return Final values of APar.
   */
  virtual mdarray<Real, dextents<Dim, 2u>> getFinalAPar() = 0;
};

} // namespace ahr