#ifndef Complex_H
#define Complex_H

#include "Core/Platform.h"

namespace PhysIKA {
template <typename Real>
class Complex
{
public:
    COMM_FUNC Complex();
    COMM_FUNC explicit Complex(Real real, Real imag = Real(0));
    COMM_FUNC Real realPart() const
    {
        return m_real;
    }
    COMM_FUNC Real imagPart() const
    {
        return m_imag;
    }

    COMM_FUNC Complex<Real> conjugate() const;
    COMM_FUNC Real          norm() const;
    COMM_FUNC Real          normSquared() const;

    COMM_FUNC bool isReal() const;

    COMM_FUNC const Complex<Real> operator+(const Complex<Real>& other) const;
    COMM_FUNC const Complex<Real> operator-(const Complex<Real>& other) const;
    COMM_FUNC const Complex<Real> operator*(const Complex<Real>& other) const;
    COMM_FUNC const Complex<Real> operator/(const Complex<Real>& other) const;

    COMM_FUNC Complex<Real>& operator+=(const Complex<Real>& other);
    COMM_FUNC Complex<Real>& operator-=(const Complex<Real>& other);
    COMM_FUNC Complex<Real>& operator*=(const Complex<Real>& other);
    COMM_FUNC Complex<Real>& operator/=(const Complex<Real>& other);

    COMM_FUNC Complex<Real>& operator=(const Complex<Real>& other);

    COMM_FUNC bool operator==(const Complex<Real>& other) const;
    COMM_FUNC bool operator!=(const Complex<Real>& other) const;

    COMM_FUNC const Complex<Real> operator+(const Real& real) const;
    COMM_FUNC const Complex<Real> operator-(const Real& real) const;
    COMM_FUNC const Complex<Real> operator*(const Real& real) const;
    COMM_FUNC const Complex<Real> operator/(const Real& real) const;

    COMM_FUNC Complex<Real>& operator+=(const Real& real);
    COMM_FUNC Complex<Real>& operator-=(const Real& real);
    COMM_FUNC Complex<Real>& operator*=(const Real& real);
    COMM_FUNC Complex<Real>& operator/=(const Real& real);

    COMM_FUNC const Complex<Real> operator-(void) const;

protected:
    Real m_real;
    Real m_imag;
};

//
///    template class Complex<float>;
///     template class Complex<double>;

template <typename Real>
inline COMM_FUNC Complex<Real> acos(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> asin(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> atan(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> asinh(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> acosh(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> atanh(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> cos(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> cosh(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> exp(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> log(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> log10(const Complex<Real>&);

template <typename Real>
inline COMM_FUNC Complex<Real> pow(const Complex<Real>&, const Real&);
template <typename Real>
inline COMM_FUNC Complex<Real> pow(const Complex<Real>&, const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> pow(const Real&, const Complex<Real>&);
//
template <typename Real>
inline COMM_FUNC Complex<Real> sin(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> sinh(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> sqrt(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> tan(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> tanh(const Complex<Real>&);

template <typename Real>
inline COMM_FUNC Real arg(const Complex<Real>&);
template <typename Real>
inline COMM_FUNC Complex<Real> polar(const Real& __rho, const Real& __theta = Real(0));
}  // namespace PhysIKA

#include "Complex.inl"

#endif  // Complex_H
