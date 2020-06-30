/** -*- mode: c++ -*-
 * @file log_utils.cc
 * @author LamKamhang (Cool_Lam@outlook.com)
 * @brief A Documented file.
 * @version 1.0
 * @date Fri Dec 6 16:28:43 CST 2019
 *
 * Detailed description
 *
 *
 * @copyright Copyright (c) 2019
 */
#ifndef CHAOS_LOG_UTILS_CC
#define CHAOS_LOG_UTILS_CC

#include "log_utils.h"
#include "chaos_log_inline.h"

namespace chaos
{
  namespace utils
  {
    CHAOS_LOG_INLINE log_printer::log_printer()
      : oss({stderr, stderr, stderr, stderr, stderr, stderr, stdout, stdout})
    {};

    CHAOS_LOG_INLINE log_printer &log_printer::get_instance()
    {
      static log_printer printer_instance;
      return printer_instance;
    }

    CHAOS_LOG_INLINE FILE *log_printer::get_os(const log_level_t level)
    {
      return oss[level];
    }

    CHAOS_LOG_INLINE bool log_printer::close_os(const log_level_t level)
    {
      FILE *old = oss[level];
      bool ret = !(old != stdout && old != stderr &&
                   // if fclose failed, how to fix the bug elegantly?
                   (std::find(oss.begin(), oss.end(), old) == oss.end() && fclose(old)));
      return ret;
    }

    CHAOS_LOG_INLINE bool log_printer::set_os(const log_level_t level,
                                              const log_out_sys_t type)
    {
      static FILE *out_sys_ptrs[] = {stdout, stderr};
      bool ret = close_os(level);
      if (ret)
        oss[level] = out_sys_ptrs[type];
      return ret;
    }

    CHAOS_LOG_INLINE bool log_printer::set_os(const log_level_t level,
                                              const std::string &file_path,
                                              const std::string &mode)
    {
      FILE *tmp = fopen(file_path.c_str(), mode.c_str());
      bool ret = (tmp != nullptr && close_os(level));
      if (ret)
        oss[level] = tmp;
      else if (tmp != nullptr)
        fclose(tmp);
      return ret;
    }

    CHAOS_LOG_INLINE const char *log_level_str(const log_level_t level)
    {
      static constexpr const char* const _level_str_map[] =
        {
          "EMERG", "ALERT", "CRITI", "ERROR",
          "WARN ", "NOTICE", "INFO ", "DEBUG",
        };
      return _level_str_map[level];
    }
  }
}

#endif /* CHAOS_LOG_UTILS_CC */
