#ifndef CPTIMER
#define CPTIMER

#include <time.h>
#include <stdio.h>
#include <chrono>

namespace math {

struct cTime
{
    std::chrono::steady_clock::time_point t1, t2;

    cTime() {}

    void tick()
    {
        t1 = std::chrono::steady_clock::now();
    }

    //return value in seconds
    double tack()
    {
        t2                                      = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        return time_span.count();
    }

    double tack(const char* str)
    {
        t2                                      = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        printf("%s %f ms\n", str, time_span.count() * 1000);
        return time_span.count();
    }
};

}  // namespace math
#endif