#ifndef STACK_H
#define STACK_H

#include "Core/Platform.h"

namespace PhysIKA {
/**
     * @brief Be aware do not use this structure on GPU if the data size is large.
     * 
     * @tparam T 
     */
template <typename T>
class Stack
{
public:
    using iterator = T*;

    COMM_FUNC Stack();

    COMM_FUNC iterator find(T val);

    COMM_FUNC inline iterator begin()
    {
        return data;
    };

    COMM_FUNC inline iterator end()
    {
        return data + m_size;
    }

    COMM_FUNC void clear();

    COMM_FUNC int size();

    COMM_FUNC iterator insert(T val);
    COMM_FUNC bool     empty();

    COMM_FUNC int  erase(const T val);
    COMM_FUNC void erase(iterator val_ptr);

private:
    int m_size = 0;
};
}  // namespace PhysIKA

#include "Stack.inl"

#endif  // STACK_H
