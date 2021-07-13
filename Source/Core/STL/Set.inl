#include "STLMacro.h"
#include "Core/Utility/SimpleMath.h"
#include <glm/glm.hpp>

namespace PhysIKA {
template <typename T>
COMM_FUNC Set<T>::Set()
    : STLBuffer()
{
}

template <typename T>
COMM_FUNC T* Set<T>::find(T val)
{
    int ind = leftBound(val, m_startLoc, m_size);

    return ind >= m_size || m_startLoc[ind] != val ? nullptr : m_startLoc + ind;
}

template <typename T>
COMM_FUNC T* Set<T>::insert(T val)
{
    //return nullptr if the data buffer is full
    if (m_size >= m_maxSize)
        return nullptr;

    //return the index of the first element that is equal to or greater than val
    int ind = leftBound(val, m_startLoc, m_size);

    if (ind == m_size)
    {
        m_startLoc[ind] = val;
        m_size++;

        return m_startLoc + ind;
    };

    //return the original address if val is found
    if (m_startLoc[ind] == val)
        return m_startLoc + ind;
    else
    {
        //if found, move all element backward
        for (int j = m_size; j > ind; j--)
        {
            m_startLoc[j] = m_startLoc[j - 1];
        }

        //insert val into location ind.
        m_startLoc[ind] = val;
        m_size++;

        return m_startLoc + ind;
    }
}

template <typename T>
COMM_FUNC void Set<T>::clear()
{
    m_size = 0;
}

template <typename T>
COMM_FUNC size_t Set<T>::size()
{
    return m_size;
}

template <typename T>
COMM_FUNC size_t Set<T>::count(T val)
{
    return find(val) ? 1 : 0;
}

template <typename T>
COMM_FUNC bool Set<T>::empty()
{
    return m_startLoc == nullptr;
}
}  // namespace PhysIKA
