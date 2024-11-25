#pragma once

#include "cilk.hpp"

#include <algorithm>
#include <limits>

namespace detail {
template <typename T> void sum_init(void *v) { *reinterpret_cast<T *>(v) = T{}; }

template <typename T> void sum_reduce(void *v, void *v2) {
  T *a = reinterpret_cast<T *>(v), *b = reinterpret_cast<T *>(v2);
  *a += *b;
}

template <typename T> void max_init(void *v) {
  *reinterpret_cast<T *>(v) = std::numeric_limits<T>::min();
}

template <typename T> void max_reduce(void *v, void *v2) {
  T *a = reinterpret_cast<T *>(v), *b = reinterpret_cast<T *>(v2);
  *a = std::max(*a, *b);
}
} // namespace detail

template <typename T> using SumReducer = T cilk_reducer(detail::sum_init<T>, detail::sum_reduce<T>);
template <typename T> using MaxReducer = T cilk_reducer(detail::max_init<T>, detail::max_reduce<T>);
