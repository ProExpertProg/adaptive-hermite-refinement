#include "grid.hpp"

#include "debug.hpp"
#include "util.hpp"

#include "CachedExponentials.hpp"
#include "Exponentials.hpp"

#include <gtest/gtest.h>

namespace ahr::exp {

template <typename Exp, typename ExpT> class TestExps : public ::testing::Test {
protected:
  Grid grid{5, 32, 64};
  Exp exp{grid};
  ExpT exp_t{grid};

  void check()
    requires exp::space_like<Exp>
  {
    grid.for_each_kxky([&](Dim kx, Dim ky) {
      EXPECT_THAT(exp(kx, ky), AllClose(exp_t(kx, ky), 1e-16, 1e-15)); // prevent oneline
    });
  }

  void check()
    requires exp::moment_like<Exp>
  {
    for (int m = 0; m < grid.M; ++m) {
      EXPECT_THAT(exp(m), AllClose(exp_t(m), 1e-16, 1e-15));
    }
  }

  void test() {
    Real dt = 1.0;
    HyperCoefficients hyper = HyperCoefficients::calculate(dt, grid);

    exp.update(hyper, dt);
    exp_t.update(hyper, dt);

    this->check();
    this->check(); // Run again

    dt *= 1.2;
    hyper = HyperCoefficients::calculate(dt, grid);
    exp.update(hyper, dt);
    exp_t.update(hyper, dt);

    this->check();

    // Update with the same dt
    exp.update(hyper, dt);
    exp_t.update(hyper, dt);
    this->check();

    // Old dt again
    exp.update(hyper, 1.0);
    exp_t.update(hyper, 1.0);
    this->check();
  }
};

/// This struct contains a type definition for the appropriate Cached* class for exp.
/// It's specialized for space-like/moment-like so they can all use the same test suite.
template <typename Exp> struct cached;

template <space_like Exp> struct cached<Exp> {
  using type = CachedKXKY<Exp>;
};

template <moment_like Exp> struct cached<Exp> {
  using type = CachedM<Exp>;
};

template <typename Exp> using TestCachedExps = TestExps<Exp, typename cached<Exp>::type>;

using Types = ::testing::Types<Eta, Nu, NuG, GM>;
TYPED_TEST_SUITE(TestCachedExps, Types);

TYPED_TEST(TestCachedExps, Exps) { this->test(); }

} // namespace ahr::exp
