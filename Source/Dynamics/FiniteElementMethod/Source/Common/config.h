/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: timing helper
 * @version    : 1.0
 */
#ifndef PhysIKA_CONFIG_H
#define PhysIKA_CONFIG_H
#include <chrono>
#include <iostream>
#include <list>
#include <string>
#include <assert.h>

namespace PhysIKA {

//TIMING
class TIMING
{
public:
    static void begin()
    {
        starts_.push_back(std::chrono::system_clock::now());
    }
    static double end(const std::string& info, const bool if_print = true)
    {
        assert(!starts_.empty());

        auto   end      = std::chrono::system_clock::now();
        auto   duration = std::chrono::duration_cast<std::chrono::microseconds>(end - starts_.back());
        double res      = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
        if (if_print)
            std::cout << info << " cost " << res << "seoncds" << std::endl;
        starts_.pop_back();
        return res;
    }

private:
    static std::list<std::chrono::system_clock::time_point> starts_;
};

#define __TIME_BEGIN__ \
    TIMING::begin();

#define __TIME_END_1__(_str) \
    TIMING::end(_str);

#define __TIME_END_2__(_str, _print) \
    TIMING::end(_str, _print);

#define __GET_THIRD_ARG__(arg1, arg2, arg3, ...) arg3
#define __TIME_END_CHOOSER__(...) \
    __GET_THIRD_ARG__(__VA_ARGS__, __TIME_END_2__, __TIME_END_1__, )
#define __TIME_END__(...)             \
    __TIME_END_CHOOSER__(__VA_ARGS__) \
    (__VA_ARGS__)

}  // namespace PhysIKA
#endif
