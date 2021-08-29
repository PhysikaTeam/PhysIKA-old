#pragma once

#include <chrono>

namespace pbal {

class Timer
{
public:
    Timer()
    {
        _startingPoint = _clock.now();
    }

    double durationInSeconds() const
    {
        auto end   = std::chrono::steady_clock::now();
        auto count = std::chrono::duration_cast<std::chrono::microseconds>(
                         end - _startingPoint)
                         .count();
        return count / 1000000.0;
    }

    void reset()
    {
        _startingPoint = _clock.now();
    }

private:
    std::chrono::steady_clock             _clock;
    std::chrono::steady_clock::time_point _startingPoint;
};

}  // namespace pbal