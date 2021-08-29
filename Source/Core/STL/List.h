#ifndef LIST_H
#define LIST_H

#include "Core/Platform.h"
#include "STLBuffer.h"

namespace PhysIKA {
/**
     * @brief Be aware do not use this structure on GPU if the data size is large.
     * 
     * @tparam T 
     */
template <typename T>
class List : public STLBuffer<T>
{
public:
    using iterator = T*;

    COMM_FUNC List();

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

    COMM_FUNC int size();

    COMM_FUNC inline iterator insert(T val);
    COMM_FUNC inline iterator atomicInsert(T val);

    COMM_FUNC bool empty();

private:
    int m_size = 0;
};
}  // namespace PhysIKA

#include "List.inl"

#endif  // LIST_H
