#pragma once

#ifndef CILK_ENABLED
#define cilk_for for
#define cilk_spawn /* empty */
#define cilk_sync  /* empty */
#define cilk_scope /* empty */
#define cilk_reducer(init, reduce) /* empty */

#else
#include <cilk/cilk.h>
#endif
