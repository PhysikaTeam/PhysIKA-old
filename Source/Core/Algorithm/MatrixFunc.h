#pragma once
#include "Core/Platform.h"
#include "Core/Matrix.h"

namespace PhysIKA {
template <typename Real, int Dim>
COMM_FUNC void polarDecomposition(const SquareMatrix<Real, Dim>& A, SquareMatrix<Real, Dim>& R, SquareMatrix<Real, Dim>& U, SquareMatrix<Real, Dim>& D, SquareMatrix<Real, Dim>& V);

template <typename Real, int Dim>
COMM_FUNC void polarDecomposition(const SquareMatrix<Real, Dim>& A, SquareMatrix<Real, Dim>& R, SquareMatrix<Real, Dim>& U, SquareMatrix<Real, Dim>& D);

template <typename Real, int Dim>
COMM_FUNC void polarDecomposition(const SquareMatrix<Real, Dim>& M, SquareMatrix<Real, Dim>& R, Real tolerance);
}  // namespace PhysIKA

#include "MatrixFunc.inl"