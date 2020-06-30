/** -*- mode: c++ -*-
 * @file assert_utils.h
 * @author LamKamhang (Cool_Lam@outlook.com)
 * @brief A Documented file.
 * @version 1.0
 * @date Mon Nov 18 10:59:29 CST 2019
 *
 * Detailed description
 *
 *
 * @copyright Copyright (c) 2019
 */
#ifndef CHAOS_ASSERT_UTILS_H
#define CHAOS_ASSERT_UTILS_H

#include "log_utils.h"

#ifdef ASSERT_EXT_FLAG
# define assert_ext(expr, event, ...)           \
  if (!(expr))                                  \
    {                                           \
      error_msg_ext("`(%s)` failed. "#event,    \
                    #expr, ##__VA_ARGS__);      \
      exit(0);                                  \
    }

#else
# define assert_ext(expr, event, ...)
#endif /* ASSERT_EXT_FLAG */

#endif /* CHAOS_ASSERT_UTILS_H */
