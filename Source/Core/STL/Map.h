#ifndef MAP_H
#define MAP_H

#include "Core/Platform.h"
#include "Pair.h"

namespace PhysIKA {
/**
     * @brief Be aware do not use this structure on GPU if the data size is large.
     * 
     * @tparam T 
     */
template <typename MKey, typename T>
class Map
{
public:
    using iterator = Pair<MKey, T>*;

    COMM_FUNC Map();

    COMM_FUNC void reserve(Pair<MKey, T>* buf, int maxSize)
    {
        m_pairs   = buf;
        m_maxSize = maxSize;
    }

    COMM_FUNC iterator find(MKey key);

    COMM_FUNC inline iterator begin()
    {
        return m_pairs;
    };

    COMM_FUNC inline iterator end()
    {
        return m_pairs + m_size;
    }

    COMM_FUNC void clear();

    COMM_FUNC int size();

    COMM_FUNC iterator insert(Pair<MKey, T> pair);
    COMM_FUNC bool     empty();

    COMM_FUNC int  erase(const T val);
    COMM_FUNC void erase(iterator val_ptr);

private:
    int m_size = 0;

    Pair<MKey, T>* m_pairs   = nullptr;
    int            m_maxSize = 0;
};
}  // namespace PhysIKA

#include "Map.inl"

#endif  // MAP_H
