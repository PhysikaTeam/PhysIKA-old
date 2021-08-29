#ifndef MultiMap_H
#define MultiMap_H

#include "Core/Platform.h"
#include "Pair.h"

namespace PhysIKA {
/**
     * @brief An CPU/GPU implementation of the standard multimap suitable for small-size data
     * 
     * Be aware do not use this structure if the data size is large,
     * because the computation complexity is O(n^2) for some specific situation.
     * 
     * All elements are organized in non-descending order.
     * 
     * @tparam T 
     */
template <typename MKey, typename T>
class MultiMap
{
public:
    using iterator = Pair<MKey, T>*;

    COMM_FUNC MultiMap();

    COMM_FUNC void reserve(Pair<MKey, T>* buf, int maxSize)
    {
        m_startLoc = buf;
        m_maxSize  = maxSize;
    }

    COMM_FUNC iterator find(MKey key);

    COMM_FUNC inline iterator begin()
    {
        return m_startLoc;
    };

    COMM_FUNC inline iterator end()
    {
        return m_startLoc + m_size;
    }

    COMM_FUNC void clear();

    COMM_FUNC size_t size();
    COMM_FUNC size_t count(MKey key);

    COMM_FUNC inline T&       operator[](MKey key);
    COMM_FUNC inline const T& operator[](MKey key) const;

    COMM_FUNC iterator insert(Pair<MKey, T> pair);
    COMM_FUNC bool     empty();

private:
    size_t m_size = 0;

    Pair<MKey, T>* m_startLoc = nullptr;
    int            m_maxSize  = 0;
};

}  // namespace PhysIKA

#include "MultiMap.inl"

#endif  // MultiSet_H
