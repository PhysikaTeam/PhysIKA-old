#ifndef SET_H
#define SET_H

#include "Core/Platform.h"
#include "STLBuffer.h"

namespace PhysIKA {
/**
     * @brief An CPU/GPU implementation of the standard set suitable for small-size data
     * 
     * Be aware do not use this structure if the data size is large.
     * The computation complexity is O(n^2) for some specific situation.
     * All elements are organized in ascending order
     * 
     * @tparam T 
     */
template <typename T>
class Set : public STLBuffer<T>
{
public:
    COMM_FUNC Set();

    COMM_FUNC T* find(T val);

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

    COMM_FUNC T*   insert(T val);
    COMM_FUNC bool empty();

    COMM_FUNC int  erase(const T val);
    COMM_FUNC void erase(T* val_ptr);

private:
    size_t m_size = 0;
};

}  // namespace PhysIKA

#include "Set.inl"

#endif  // SET_H
