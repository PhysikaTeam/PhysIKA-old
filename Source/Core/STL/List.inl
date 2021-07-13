#include "Core/Utility/SimpleMath.h"
#include <glm/glm.hpp>

namespace PhysIKA {
template <typename T>
COMM_FUNC List<T>::List()
{
}

template <typename T>
COMM_FUNC T* List<T>::find(T val)
{
    return nullptr;
}

template <typename T>
COMM_FUNC T* List<T>::insert(T val)
{
    m_startLoc[m_size] = val;
    m_size++;

    return m_startLoc + m_size - 1;
    ;
}

template <typename T>
COMM_FUNC T* List<T>::atomicInsert(T val)
{
    int index         = atomicAdd(&m_size, 1);
    m_startLoc[index] = val;

    return m_startLoc + index;
}

template <typename T>
COMM_FUNC void List<T>::clear()
{
    m_size = 0;
}

template <typename T>
COMM_FUNC int List<T>::size()
{
    return m_size;
}

template <typename T>
COMM_FUNC bool List<T>::empty()
{
    return m_size == 0;
}
}  // namespace PhysIKA
