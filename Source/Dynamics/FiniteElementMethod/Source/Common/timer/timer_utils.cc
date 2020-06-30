/** -*- mode: c++ -*-
 * @file timer_utils.cc
 * @author LamKamhang (Cool_Lam@outlook.com)
 * @brief A Documented file.
 * @version 1.1
 * @date Fri Dec 6 16:17:20 CST 2019
 *
 * Detailed description
 *
 *
 * @copyright Copyright (c) 2019
 */
#ifndef CHAOS_TIMER_UTILS_CC
#define CHAOS_TIMER_UTILS_CC

#include <limits>

#include "timer_utils.h"
#include "chaos_timer_inline.h"

namespace chaos
{
  namespace utils
  {
    CHAOS_TIMER_INLINE const char *timer_scale_units_str (timer_scale_t scale)
    {
      static constexpr const char* const units[] = {
        "h", "min", "s", "ms", "μs", "ns", "custom units"
      };
      return units[scale];
    }

    ///////////////////////////////////////////////////////////////////////////
    //                       stopwatch_t implementation                      //
    ///////////////////////////////////////////////////////////////////////////
    CHAOS_TIMER_INLINE stopwatch_t::stopwatch_t(timer_scale_t scale)
      : tp_begin(std::chrono::steady_clock::now())
      , tp_end(tp_begin)
      , scale(scale)
    { reset(); }

    CHAOS_TIMER_INLINE void stopwatch_t::start()
    {
      warn_msg_cond(!isstop, "the stopwatch has started. Now it will reset it");
      tp_begin = std::chrono::steady_clock::now();
      isstop = false;
    }

    CHAOS_TIMER_INLINE void stopwatch_t::reset()
    {
      cnt = 0;
      avg_cost = 0;
      min_cost = std::numeric_limits<double>::max() / 100;
      max_cost = 0;
      isstop = true;
    }

    CHAOS_TIMER_INLINE void stopwatch_t::reset(timer_scale_t scale)
    { this->scale = scale; reset(); }

    CHAOS_TIMER_INLINE void stopwatch_t::set_scale(timer_scale_t scale)
    { this->scale = scale; }

    CHAOS_TIMER_INLINE std::string
    stopwatch_t::get_avg(const std::string &event) const
    { return get_avg(event, timer_scale_units_str(scale)); }

    CHAOS_TIMER_INLINE std::string
    stopwatch_t::get_min(const std::string &event) const
    { return get_min(event, timer_scale_units_str(scale)); }

    CHAOS_TIMER_INLINE std::string
    stopwatch_t::get_max(const std::string &event) const
    { return get_max(event, timer_scale_units_str(scale)); }

    CHAOS_TIMER_INLINE std::string
    stopwatch_t::get_avg(const std::string &event,
                         const std::string &unit) const
    { return event + " avg costs " + std::to_string(avg_cost) + " " + unit; }

    CHAOS_TIMER_INLINE std::string
    stopwatch_t::get_min(const std::string &event,
                         const std::string &unit) const
    { return event + " min costs " + std::to_string(min_cost) + " " + unit; }

    CHAOS_TIMER_INLINE std::string
    stopwatch_t::get_max(const std::string &event,
                         const std::string &unit) const
    { return event + " max costs " + std::to_string(max_cost) + " " + unit; }

    CHAOS_TIMER_INLINE double stopwatch_t::get_avg() const
    { return avg_cost; }

    CHAOS_TIMER_INLINE double stopwatch_t::get_min() const
    { return min_cost; }

    CHAOS_TIMER_INLINE double stopwatch_t::get_max() const
    { return max_cost; }

    CHAOS_TIMER_INLINE timer_scale_t stopwatch_t::get_scale() const
    { return scale; }

    CHAOS_TIMER_INLINE bool stopwatch_t::is_stop() const
    { return isstop; }

    ///////////////////////////////////////////////////////////////////////////
    //                    stopwatch manager implementation                   //
    ///////////////////////////////////////////////////////////////////////////
    CHAOS_TIMER_INLINE stopwatch_manager_t& stopwatch_manager_t::get_instance ()
    { static stopwatch_manager_t instance; return instance; }

    CHAOS_TIMER_INLINE stopwatch_id_t
    stopwatch_manager_t::create (timer_scale_t scale)
    {
      std::shared_ptr<stopwatch_t> stw = std::make_shared<stopwatch_t>(scale);
      stopwatch_id_t id = std::hash<std::shared_ptr<stopwatch_t> >{}(stw);
      stwmaps.insert({id, stw});
      return id;
    }

    CHAOS_TIMER_INLINE void stopwatch_manager_t::remove (stopwatch_id_t id)
    { stwmaps.erase(id); }

    CHAOS_TIMER_INLINE std::shared_ptr<stopwatch_t>
    stopwatch_manager_t::get (stopwatch_id_t id)
    {
      if (stwmaps.count(id) != 0)
      return stwmaps[id];
      else
      {
        std::shared_ptr<stopwatch_t> stw = std::make_shared<stopwatch_t>();
        stwmaps.insert({id, stw});
        return stw;
      }
    }

#ifdef USE_STOPWATCH_HELPER_FLAG
    CHAOS_TIMER_INLINE stopwatch_id_t STW_CREATE (timer_scale_t scale)
    { return stopwatch_manager_t::get_instance().create(scale); }

    CHAOS_TIMER_INLINE stopwatch_id_t STW_CREATE_START (timer_scale_t scale)
    {
      stopwatch_id_t id = stopwatch_manager_t::get_instance().create(scale);
      STW_START (id);
      return id;
    }

    CHAOS_TIMER_INLINE void STW_START (const std::string &token)
    {
      STW_START(std::hash<std::string>{}(token));
    }

    CHAOS_TIMER_INLINE void STW_START (const stopwatch_id_t &id)
    {
      const auto & stw = stopwatch_manager_t::get_instance().get(id);
      stw->start();
    }

    CHAOS_TIMER_INLINE void STW_RESET
    (const std::string &token, timer_scale_t scale)
    {
      STW_RESET(std::hash<std::string>{}(token), scale);
    }
    CHAOS_TIMER_INLINE void STW_RESET
    (const stopwatch_id_t &id, timer_scale_t scale)
    {
      const auto & stw = stopwatch_manager_t::get_instance().get(id);
      stw->reset (scale);
    }

    CHAOS_TIMER_INLINE void STW_RESET_START
    (const std::string &token, timer_scale_t scale)
    {
      STW_RESET_START(std::hash<std::string>{}(token), scale);
    }

    CHAOS_TIMER_INLINE void STW_RESET_START
    (const stopwatch_id_t &id, timer_scale_t scale)
    {
      const auto & stw = stopwatch_manager_t::get_instance().get(id);
      stw->reset (scale);
      stw->start ();
    }

#else
    CHAOS_TIMER_INLINE stopwatch_id_t STW_CREATE
    (timer_scale_t __attribute__((__unused__)) scale)
    { return 0; }

    CHAOS_TIMER_INLINE stopwatch_id_t STW_CREATE_START
    (timer_scale_t __attribute__((__unused__)) scale)
    { return 0; }

    CHAOS_TIMER_INLINE void STW_START
    (const std::string __attribute__((__unused__)) &token){}
    CHAOS_TIMER_INLINE void STW_START
    (const stopwatch_id_t __attribute__((__unused__)) &id){}

    CHAOS_TIMER_INLINE void STW_RESET
    (const std::string __attribute__((__unused__)) &token,
     timer_scale_t __attribute__((__unused__)) scale){}
    CHAOS_TIMER_INLINE void STW_RESET
    (const stopwatch_id_t __attribute__((__unused__)) &id,
     timer_scale_t __attribute__((__unused__)) scale){}

    CHAOS_TIMER_INLINE void STW_RESET_START
    (const std::string __attribute__((__unused__)) &token,
     timer_scale_t __attribute__((__unused__)) scale){}
    CHAOS_TIMER_INLINE void STW_RESET_START
    (const stopwatch_id_t __attribute__((__unused__)) &id,
     timer_scale_t __attribute__((__unused__)) scale){}
#endif /*  USE_STOPWATCH_HELPER_FLAG */
  } // utils
} // chaos

#endif /* CHAOS_TIMER_UTILS_CC */
