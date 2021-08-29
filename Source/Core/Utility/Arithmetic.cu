#include "Arithmetic.h"
#include "Function1Pt.h"
#include "Function2Pt.h"

namespace PhysIKA {

template <typename T>
Arithmetic<T>::Arithmetic(int n)
    : m_reduce(NULL)
{
    m_reduce = Reduction<T>::Create(n);
    m_buf.resize(n);
}

template <typename T>
Arithmetic<T>::~Arithmetic()
{
    if (m_reduce != NULL)
    {
        delete m_reduce;
    }
    m_buf.release();
}

template <typename T>
Arithmetic<T>* Arithmetic<T>::Create(int n)
{
    return new Arithmetic<T>(n);
}

template <typename T>
T Arithmetic<T>::Dot(DeviceArray<T>& xArr, DeviceArray<T>& yArr)
{
    Function2Pt::multiply(m_buf, xArr, yArr);
    return m_reduce->accumulate(m_buf.getDataPtr(), m_buf.size());
}

}  // namespace PhysIKA