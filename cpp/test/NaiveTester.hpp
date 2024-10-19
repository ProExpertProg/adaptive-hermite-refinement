#pragma once
#include "Naive.hpp"
#include "util.hpp"

#include <gtest/gtest.h>

namespace ahr {

namespace utils {
// Utilities
template <typename... Args, typename... ArgsCat>
auto tuple_append(std::tuple<Args...> tuple, ArgsCat &&...args) {
  return std::tuple_cat(tuple, std::make_tuple(std::forward<ArgsCat>(args)...));
}

template <typename Tup, typename... Args>
using tuple_append_t = decltype(tuple_append(std::declval<Tup>(), std::declval<Args>()...));

/// Convert double to short string (2 decimal places)
static std::string d_to_str(double d) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << d;
  auto r = oss.str();
  std::ranges::replace(r, '.', '_');
  return r;
}
} // namespace utils

template <typename T>
concept param_like = requires(T t, typename T::Tuple tup) {
                       std::tuple_size_v<typename T::Tuple>;
                       { T{tup} } -> std::same_as<T>;
                       { t.to_param_str() } -> std::same_as<std::string>;
                     };

struct NaiveParam {
  Dim M, X, N;

  /// M, X, N
  using Tuple = std::tuple<Dim, Dim, Dim>;

  explicit NaiveParam(Tuple t) : M(std::get<0>(t)), X(std::get<1>(t)), N(std::get<2>(t)) {}

  std::string to_param_str() const {
    std::ostringstream oss;
    oss << "M" << M << "_X" << X << "_N" << N;
    return oss.str();
  }
};

template <param_like BaseParam> struct WithDiffusion : BaseParam {
  Real res, nu;

  using BaseTuple = typename BaseParam::Tuple;
  static constexpr auto BaseSize = std::tuple_size_v<BaseTuple>;

  using Tuple = utils::tuple_append_t<BaseTuple, Real, Real>;

  explicit WithDiffusion(Tuple t)
      : BaseParam(slice<0, BaseSize>(t)), res(std::get<BaseSize>(t)),
        nu(std::get<BaseSize + 1>(t)) {}

  std::string nu_str() const { return utils::d_to_str(nu); }

  std::string res_str() const { return utils::d_to_str(res); }

  std::string to_param_str() const {
    return BaseParam::to_param_str() + "_nu" + nu_str() + "_res" + res_str();
  }
};

template <param_like BaseParam> struct WithHyperDiffusion : BaseParam {
  Real hyper_coef_g, hyper_coef, hyperm_coef;
  using BaseTuple = typename BaseParam::Tuple;
  static constexpr auto BaseSize = std::tuple_size_v<BaseTuple>;

  using Tuple = utils::tuple_append_t<BaseTuple, Real, Real, Real>;

  explicit WithHyperDiffusion(Tuple t)
      : BaseParam(slice<0, BaseSize>(t)), hyper_coef_g(std::get<BaseSize>(t)),
        hyper_coef(std::get<BaseSize + 1>(t)), hyperm_coef(std::get<BaseSize + 2>(t)) {}

  std::string hyper_g_string() const { return utils::d_to_str(hyper_coef_g); }
  std::string hyper_string() const { return utils::d_to_str(hyper_coef); }
  std::string hyperm_string() const { return utils::d_to_str(hyperm_coef); }

  std::string to_param_str() const {
    return BaseParam::to_param_str() + "_hg" + hyper_g_string() + "_h" + hyper_string() + "_hm" +
           hyperm_string();
  }
};

template <param_like BaseParam> struct WithEquilibrium : BaseParam {
  std::string equilibrium;

  using BaseTuple = typename BaseParam::Tuple;
  static constexpr auto BaseSize = std::tuple_size_v<BaseTuple>;

  using Tuple = utils::tuple_append_t<BaseTuple, std::string>;

  explicit WithEquilibrium(Tuple t)
      : BaseParam(slice<0, BaseSize>(t)), equilibrium(std::get<BaseSize>(t)) {}

  std::string to_param_str() const { return BaseParam::to_param_str() + "_" + equilibrium; }
};

// using TesterParam = WithDiffusion<NaiveParam>;

class TesterWithOutput : public ::testing::Test {
protected:
  // Ignore output
  std::ostringstream out{};

  void TearDown() override {
    if (HasFailure()) { std::cout << "============\nFull output:\n" << out.str() << std::endl; }
  }
};

template <class Param>
class NaiveTester : public TesterWithOutput, public testing::WithParamInterface<Param> {
public:
  class Printer {
  public:
    std::string operator()(const testing::TestParamInfo<Param> &info) const {
      return info.param.to_param_str();
    }
  };
};

} // namespace ahr
