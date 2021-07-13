#ifndef MultiSet_H
#define MultiSet_H

#include "Core/Platform.h"
#include "STLBuffer.h"

namespace PhysIKA {
/**
     * @brief An CPU/GPU implementation of the standard multiset suitable for small-size data
     * 
     * Be aware do not use this structure if the data size is large,
     * because the computation complexity is O(n^2) for some specific situation.
     * 
     * All elements are organized in non-descending order.
     * 
     * @tparam T 
     */
template <typename T>
class MultiSet : public STLBuffer<T>
{
public:
    using iterator = T*;

    COMM_FUNC MultiSet();

    COMM_FUNC iterator find(T val);

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
    COMM_FUNC size_t count(T val);

    COMM_FUNC iterator insert(T val);
    COMM_FUNC bool     empty();

    COMM_FUNC int  erase(const T val);
    COMM_FUNC void erase(iterator val_ptr);

private:
    int m_size = 0;
};

}  // namespace PhysIKA

#include "MultiSet.inl"

#endif  // MultiSet_H
