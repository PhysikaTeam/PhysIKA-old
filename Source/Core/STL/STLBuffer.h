#ifndef STLBUFFER_H
#define STLBUFFER_H

#include "Core/Platform.h"

namespace PhysIKA {
/**
     * @brief Be aware do not use this structure on GPU if the data size is large.
     * 
     * @tparam T 
     */
template <typename T>
class STLBuffer
{
public:
    using iterator = T*;

    COMM_FUNC STLBuffer(){};

    COMM_FUNC void reserve(T* beg, int buffer_size)
    {
        m_startLoc = beg;
        m_maxSize  = buffer_size;
    }

    COMM_FUNC int max_size()
    {
        return m_maxSize;
    }

    //         COMM_FUNC inline T& operator[] (unsigned int) { return m_startLoc[i]; }
    //         COMM_FUNC inline const T& operator[] (unsigned int) const { return m_startLoc[i]; }

protected:
    COMM_FUNC inline T* bufferEnd()
    {
        return m_startLoc + m_maxSize;
    }

    int m_maxSize = 0;

    T* m_startLoc = nullptr;
};
}  // namespace PhysIKA

#endif  // STLBUFFER_H
