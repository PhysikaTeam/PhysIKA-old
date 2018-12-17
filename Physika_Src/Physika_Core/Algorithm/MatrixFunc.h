#pragma once
#include "Physika_Core/Platform.h"
#include "Physika_Core/Matrices/matrix.h"

namespace Physika
{
	template<typename Real, int Dim>
	COMM_FUNC void polarDecomposition(const SquareMatrix<Real, Dim> &A, SquareMatrix<Real, Dim> &R, SquareMatrix<Real, Dim> &U, SquareMatrix<Real, Dim> &D);

	template<typename Real, int Dim>
	COMM_FUNC void polarDecomposition(const SquareMatrix<Real, Dim> &M, SquareMatrix<Real, Dim> &R, Real tolerance);
}

#include "MatrixFunc.inl"