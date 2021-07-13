#ifndef PAIR_H
#define PAIR_H

#include "Core/Platform.h"

namespace PhysIKA {
template <typename Key, typename T>
class Pair
{
public:
    COMM_FUNC Pair(){};
    COMM_FUNC Pair(Key key, T val)
    {
        first  = key;
        second = val;
    }

    COMM_FUNC inline bool operator>=(const Pair& other) const
    {
        return first >= other.first;
    }

    COMM_FUNC inline bool operator>(const Pair& other) const
    {
        return first > other.first;
    }

    COMM_FUNC inline bool operator<=(const Pair& other) const
    {
        return first <= other.first;
    }

    COMM_FUNC inline bool operator<(const Pair& other) const
    {
        return first < other.first;
    }

    COMM_FUNC inline bool operator==(const Pair& other) const
    {
        return first == other.first;
    }

    COMM_FUNC inline bool operator!=(const Pair& other) const
    {
        return first != other.first;
    }

public:
    Key first;
    T   second;
};
}  // namespace PhysIKA

#endif  // PAIR_H
