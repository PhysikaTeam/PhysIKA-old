/** -*- mode: c++ -*-
 * @file log_utils.h
 * @author LamKamhang (Cool_Lam@outlook.com)
 * @brief A Documented file.
 * @version 1.0
 * @date Mon Nov 11 17:17:53 CST 2019
 *
 * Detailed description
 *
 *
 * @copyright Copyright (c) 2019
 */

#ifndef CHAOS_LOG_UTILS_H
#define CHAOS_LOG_UTILS_H
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>


namespace chaos
{
  namespace utils
  {
    enum log_level_t
    {
      // the system is unusable.
      EMERG = 0,
      // Actions that must be taken care of immediately.
      ALERT,
      // Critical conditions.
      CRITICAL,
      // Noncritical error conditions.
      ERROR,
      // Warning conditions that should be taken care of.
      WARN,
      // Normal, but significant events.
      NOTICE,
      // Informational messages that require no action.
      INFO,
      // log debugging messages.
      DEBUG,
    };

    enum log_out_sys_t
    {
      STDOUT = 0,
      STDERR,
    };

    class log_printer
    {
#ifdef CHAOS_SINGLE_PRINTER_FLAG
    private:
      log_printer();
      log_printer(const log_printer &other) = delete;
      log_printer(log_printer &&other) = delete;
      log_printer &operator=(const log_printer &other) = delete;
#else
    public:
      log_printer();
#endif

    public:
      static log_printer &get_instance();

    public:
      FILE *get_os(const log_level_t level);
      bool set_os(const log_level_t level,
                  const std::string &file_path,
                  const std::string &mode = "w");
      bool set_os(const log_level_t level,
                  const log_out_sys_t type);

    private:
      bool close_os(const log_level_t level);
    private:
      std::vector<FILE *> oss;
    };

    const char *log_level_str(const log_level_t level);

    template<typename T>
    inline const std::string to_str(const T &item,
                                    const std::string &head = "",
                                    const std::string &tail = "")
    {
      std::ostringstream ss;
      ss << head << item << tail;
      return ss.str();
    }
  }
}

#if !defined(CHAOS_UTILS_LIBRARY_FLAG) && !defined(CHAOS_LOG_LIBRARY_FLAG)
# include "log_utils.cc"
#endif

#include "log_macro.imp"
#endif /* CHAOS_LOG_UTILS_H */
