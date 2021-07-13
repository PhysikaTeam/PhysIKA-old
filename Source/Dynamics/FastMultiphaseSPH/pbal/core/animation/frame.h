#pragma once

namespace pbal {

struct Frame final
{
    int    index = 0;
    double dt    = 1.0 / 60.0;

    Frame() {}

    Frame(int newIndex, double newDt)
        : index(newIndex), dt(newDt)
    {
    }

    //! Time in seconds
    double time() const
    {
        return index * dt;
    }

    void advance()
    {
        ++index;
    }

    void advance(int delta)
    {
        index += delta;
    }

    Frame& operator++()
    {
        advance();
        return (*this);
    }

    Frame operator++(int)
    {
        Frame result = *this;
        advance();
        return result;
    }
};

}  // namespace pbal