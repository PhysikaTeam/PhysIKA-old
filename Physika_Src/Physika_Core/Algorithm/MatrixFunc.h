#pragma once
#include "Platform.h"
#include "Physika_Core/Matrices/matrix.h"

namespace Physika
{
	template <typename Real, int Dim>
	COMM_FUNC SquareMatrix<Real, Dim> inverse(const SquareMatrix<Real, Dim> mat);

	template<typename Real, int Dim>
	COMM_FUNC void polarDecomposition(const SquareMatrix<Real, Dim> &A, SquareMatrix<Real, Dim> &R, SquareMatrix<Real, Dim> &U, SquareMatrix<Real, Dim> &D);
}

#include "MatrixFunc.inl"