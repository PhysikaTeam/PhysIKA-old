/** -*- mode: c++ -*-
 * @file timer_utils.h
 * @author LamKamhang (Cool_Lam@outlook.com)
 * @brief A Documented file.
 * @version 1.1
 * @date Thu Nov 7 15:11:24 CST 2019
 *
 * Detailed description
 *
 *
 * @copyright Copyright (c) 2019
 */
#ifndef CHAOS_TIMER_UTILS_H
#define CHAOS_TIMER_UTILS_H

#include <chrono>
#include <string>

#include <memory>
#include <functional>
#include <map>

#include "../logger/log_utils.h"

namespace chaos
{
  namespace utils
  {
    /**
     * @brief timer_scale_t is a simple scale for the timer utils.
     *
     */
    enum timer_scale_t
      {
        TS_HOURS = 0,
        TS_MINUTES,
        TS_SECONDS,
        TS_MILLISECONDS,
        TS_MICROSECONDS,
        TS_NANOSECONDS,
        TS_CUSTOM,
      };

    /**
     * @brief this rename is for stopwatch manager
     *
     */
    using stopwatch_id_t = size_t;

    /**
     * @brief timer_scale_units_str is used to set the default unit str.
     *
     * @param scale : the enumeration of timer_scale_t
     * @return const char* : the corresponding unit for the scale.
     */
    const char *timer_scale_units_str (timer_scale_t scale);

    /**
     * @brief this class is a simple stopwatch class.
     * [note1:] it cannot start twice without any stop between the calls.
     * [note2:] it cannot stop twice without any start between the calls.
     *
     */
    class stopwatch_t
    {
      /**
       * @brief this private function is to override the global eps function.
       * this is used for set the minimum duration, and any duration less than
       * this eps will be regarded as a warning.
       *
       * @tparam T : type T is the Rep of the duration type.
       * @return constexpr T : return value is the eps.
       */
      template<typename T> inline constexpr T eps () const {return (T) 1e-2f;}
      using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
    public:
      /**
       * @brief Construct a new stopwatch t object with a default scale --> ms
       *
       * @param scale : we can set another scale, too. see about timer_scale_t.
       */
      stopwatch_t(timer_scale_t scale = TS_MILLISECONDS);

    public:
      /**
       * @brief this function will start the stopwatch.
       * keep in mind that do not start more than once without any stop.
       *
       */
      void start();

      /**
       * @brief stop function will simply stop the stopwatch and calculate
       * some costs such avg_cost, min_cost and max_cost.
       * keep in mind that do not stop more than once without any start again.
       *
       * @tparam double : the default Rep is double.
       */
      template<class Rep = double> void stop();

      /**
       * @brief override a stop function that supports a custom Period.
       *
       * [note1:] we should call get_avg with a custom unit instead of
       * the function without custom unit. (we need to call get_XXX(event, unit)).
       *
       * [note2:] we should keep the same Period among each stop until
       * this evaluation finished. For instance, we cannot call stop<Rep1, Period1>(),
       * and then call stop<Rep2, Period2>() without any reset() call.
       * this will make the costs chaos and the result is not reliable.
       *
       *
       * @tparam Rep : a custom Rep(such as int/unsigned int/double/float)
       * @tparam Period : a custom Period(like: std::ratio<num, denom>.
       *                  see https://en.cppreference.com/w/cpp/numeric/ratio/ratio for detail.)
       */
      template<class Rep, class Period> void stop();

      /**
       * @brief override a stop function, let it return the corresponding string simply.
       * the notices are the same with template<class Rep> stop().
       *
       * @param event: this is the event string.
       */
      template<class Rep = double> std::string stop (const std::string &event);

      /**
       * @brief override a stop function, let it return the corresponding string simply.
       * the notices are the same with template<class Rep, class Period> stop().
       *
       * @param event: this is the event string.
       * @param unit: this is the custom period unit.
       */
      template<class Rep, class Period> std::string stop(const std::string &event,
                                                         const std::string &unit);

      /**
       * @brief reset function will reset the stopwatch.
       * clear all the costs.
       */
      void reset();

      /**
       * @brief reset function will reset the stopwatch and reset the scale.
       * clear all the costs.
       *
       * @param scale : set a new scale to the stopwatch.
       */
      void reset(timer_scale_t scale);

      /**
       * @brief Set the scale to the stopwatch.
       * [note1:] please do not call this function during any evaluation process.
       * During that process, this function will destroy the evaluation result.
       *
       * @param scale : set a new scale to the stopwatch.
       */
      void set_scale (timer_scale_t scale);

      /**
       * @brief Get the avg value.
       *
       * @param event : the event that is evaluating.
       * @return std::string : the combind output string.
       */
      std::string get_avg(const std::string &event) const;

      /**
       * @brief Get the min object
       *
       * @param event : the event that is evaluating.
       * @return std::string : the combind output string.
       */
      std::string get_min(const std::string &event) const;

      /**
       * @brief Get the max object
       *
       * @param event : the event that is evaluating.
       * @return std::string : the combind output string.
       */
      std::string get_max(const std::string &event) const;

      /**
       * @brief Get the avg object
       *
       * @param event : the event that is evaluating.
       * @param unit : the custom unit.
       * @return std::string : the combind output string.
       */
      std::string get_avg(const std::string &event, const std::string &unit) const;

      /**
       * @brief Get the min object
       *
       * @param event : the event that is evaluating.
       * @param unit : the custom unit.
       * @return std::string : the combind output string.
       */
      std::string get_min(const std::string &event, const std::string &unit) const;

      /**
       * @brief Get the max object
       *
       * @param event : the event that is evaluating.
       * @param unit : the custom unit.
       * @return std::string : the combind output string.
       */
      std::string get_max(const std::string &event, const std::string &unit) const;

      /**
       * @brief Get the avg object
       *
       * @return double : the avg_cost.
       */
      double get_avg() const;

      /**
       * @brief Get the min object
       *
       * @return double : the min_cost
       */
      double get_min() const;

      /**
       * @brief Get the max object
       *
       * @return double : the max_cost
       */
      double get_max() const;

      /**
       * @brief Get the scale object
       *
       * @return timer_scale_t : the scale in the stopwatch.
       */
      timer_scale_t get_scale() const;

      /**
       * @brief whether the stopwatch is stop or not.
       *
       */
      bool is_stop () const;

    private:
      time_point_t tp_begin;
      time_point_t tp_end;
      timer_scale_t scale;

      unsigned long cnt;
      double avg_cost;
      double min_cost;
      double max_cost;

      bool isstop;
    };

    class stopwatch_manager_t
    {
#ifdef SINGLE_STOPWATCH_MANAGER_FLAG
    private:
      stopwatch_manager_t () = default;

      //! Copy constructor
      stopwatch_manager_t (const stopwatch_manager_t &other) = delete;

      //! Move constructor
      stopwatch_manager_t (stopwatch_manager_t &&other) = delete;

      //! Assign constructor
      stopwatch_manager_t& operator=(const stopwatch_manager_t &other) = delete;
#else
    public:
      stopwatch_manager_t () = default;
#endif

    public:
    /**
     * @brief Get the instance object
     * given a static instance of stopwatch manager.
     *
     * @return stopwatch_manager_t&
     */
      static stopwatch_manager_t& get_instance ();

    public:
      /**
       * @brief create a stopwatch and return its corresponding id.
       *
       * @param scale : customize the scale of the stopwatch.
       * @return stopwatch_id_t : the id of the stopwatch.
       */
      stopwatch_id_t create (timer_scale_t scale = TS_MILLISECONDS);

      /**
       * @brief remove the stopwatch from the manager.
       *
       * @param id : the corresponding id of the stopwatch.
       */
      void remove (stopwatch_id_t id);

      /**
       * @brief get the corresponding stopwatch using the id.
       * [note1:] if there is no stopwatch with such id, this function
       * will create a stopwatch with it, which means this function will
       * also create a new stopwatch if not such stopwatch
       *
       * @param id : the stopwatch id.
       * @return std::shared_ptr<stopwatch_t> : return the corresponding stw.
       */
      std::shared_ptr<stopwatch_t> get (stopwatch_id_t id);

    private:
      /**
       * @brief the storage of the stopwatch.
       *
       */
      std::map<stopwatch_id_t, std::shared_ptr<stopwatch_t> > stwmaps;
    };

    ///////////////////////////////////////////////////////////////////////////
    //                       STOPWATCH HELPER FUNCTION                       //
    ///////////////////////////////////////////////////////////////////////////

    /**
     * @brief create a new stopwatch and return its id.
     * [note:] the stopwatch does not start. Still need
     * to call start function to start it.
     *
     * @param scale : the stopwatch's scale.
     * @return stopwatch_id_t : the id of the stw.
     */
    stopwatch_id_t STW_CREATE (timer_scale_t scale = TS_MILLISECONDS);

    /**
     * @brief create a stopwatch and start it.
     *
     * @param scale : the stopwatch' scale
     * @return stopwatch_id_t : the id of the stw.
     */
    stopwatch_id_t STW_CREATE_START (timer_scale_t scale = TS_MILLISECONDS);

    /**
     * @brief start the stopwatch.
     *
     * @param token : string token.
     */
    void STW_START (const std::string &token);

    /**
     * @brief start the stopwatch
     *
     * @param id : the corresponding id.
     */
    void STW_START (const stopwatch_id_t &id);

    /**
     * @brief reset the stopwatch
     *
     * @param token : string token
     * @param scale : new scale
     */
    void STW_RESET
    (const std::string &token, timer_scale_t scale = TS_MILLISECONDS);

    /**
     * @brief reset the stopwatch
     *
     * @param id : the corresponding id.
     * @param scale : new scale.
     */
    void STW_RESET
    (const stopwatch_id_t &id, timer_scale_t scale = TS_MILLISECONDS);

    /**
     * @brief reset and start the stopwatch.
     *
     * @param token : string token
     * @param scale : new scale.
     */
    void STW_RESET_START
    (const std::string &token, timer_scale_t scale = TS_MILLISECONDS);

    /**
     * @brief reset and start the stopwatch
     *
     * @param id : the corresponding id
     * @param scale : new scale
     */
    void STW_RESET_START
    (const stopwatch_id_t &id, timer_scale_t scale = TS_MILLISECONDS);

    /**
     * @brief sop the stopwatch.
     *
     * @tparam double
     * @param token
     */
    template<class Rep = double>
    void STW_STOP (const std::string &token);

    /**
     * @brief stop the stopwatch
     *
     * @tparam double
     * @param id
     */
    template<class Rep = double>
    void STW_STOP (const stopwatch_id_t &id);

    /**
     * @brief stop the stopwatch
     *
     * @tparam Rep
     * @tparam Period
     * @param token
     */
    template<class Rep, class Period>
    void STW_STOP (const std::string &token);

    /**
     * @brief stop the stopwatch
     *
     * @tparam Rep
     * @tparam Period
     * @param id
     */
    template<class Rep, class Period>
    void STW_STOP (const stopwatch_id_t &id);

    /**
     * @brief end the stopwatch
     *
     * @tparam double
     * @param token
     * @param event
     */
    template<class Rep = double>
    void STW_END (const std::string &token);
    /**
     * @brief end the stopwatch
     *
     * @tparam double
     * @param token
     * @param event
     */
    template<class Rep = double>
    void STW_END (const std::string &token, const std::string &event);

    /**
     * @brief end the stopwatch
     *
     * @tparam double
     * @param id
     * @param event
     */
    template<class Rep = double>
    void STW_END (const stopwatch_id_t &id, const std::string &event);

    /**
     * @brief end the stopwatch
     *
     * @tparam Rep
     * @tparam Period
     * @param token
     * @param event
     * @param unit
     */
    template<class Rep, class Period>
    void STW_END
    (const std::string &token, const std::string &event, const std::string &unit);

    /**
     * @brief end the stopwatch
     *
     * @tparam Rep
     * @tparam Period
     * @param id
     * @param event
     * @param unit
     */
    template<class Rep, class Period>
    void STW_END
    (const stopwatch_id_t &id, const std::string &event, const std::string &unit);
  }  // utils
}  // chaos

#if !defined(CHAOS_UTILS_LIBRARY_FLAG) && !defined(CHAOS_TIMER_LIBRARY_FLAG)
# include "timer_utils.cc"
#endif

#include "timer_template.imp"

#endif /* CHAOS_TIMER_UTILS_H */
