#include "Core/Utility/SimpleMath.h"
#include <glm/glm.hpp>

namespace PhysIKA {
template <typename T>
COMM_FUNC Stack<T>::Stack()
{
}

template <typename T>
COMM_FUNC T* Stack<T>::find(T val)
{
    return leftBound(val, data, m_size);
}

template <typename T>
COMM_FUNC T* Stack<T>::insert(T val)
{
    int t = leftBound(val, data, m_size);

    if (t == INVALID)
        return nullptr;
    if (data[t] == val)
        return data + t;

    for (int j = m_size; j > t; t--)
    {
        data[j] = data[j - 1];
    }
    data[t] = val;

    return data + t;
}

template <typename T>
COMM_FUNC void Stack<T>::clear()
{
    m_size = 0;
}

template <typename T>
COMM_FUNC int Stack<T>::size()
{
    return m_size;
}

template <typename T>
COMM_FUNC bool Stack<T>::empty()
{
    return data == nullptr;
}
}  // namespace PhysIKA
