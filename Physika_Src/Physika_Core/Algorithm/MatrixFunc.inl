#include "Physika_Core/Vectors/vector.h"
#include "Physika_Core/Matrices/matrix.h"

namespace Physika
{
	template<typename Real>
	COMM_FUNC void jacobiRotate(SquareMatrix<Real, 3> &A, SquareMatrix<Real, 3> &R, int p, int q)
	{
		// rotates A through phi in pq-plane to set A(p,q) = 0
		// rotation stored in R whose columns are eigenvectors of A
		if (A(p, q) == 0.0f)
			return;

		Real d = (A(p, p) - A(q, q)) / (2.0f*A(p, q));
		Real t = 1.0f / (fabs(d) + sqrt(d*d + 1.0f));
		if (d < 0.0f) t = -t;
		Real c = 1.0f / sqrt(t*t + 1);
		Real s = t*c;
		A(p, p) += t*A(p, q);
		A(q, q) -= t*A(p, q);
		A(p, q) = A(q, p) = 0.0f;
		// transform A
		int k;
		for (k = 0; k < 3; k++) {
			if (k != p && k != q) {
				Real Akp = c*A(k, p) + s*A(k, q);
				Real Akq = -s*A(k, p) + c*A(k, q);
				A(k, p) = A(p, k) = Akp;
				A(k, q) = A(q, k) = Akq;
			}
		}
		// store rotation in R
		for (k = 0; k < 3; k++) {
			Real Rkp = c*R(k, p) + s*R(k, q);
			Real Rkq = -s*R(k, p) + c*R(k, q);
			R(k, p) = Rkp;
			R(k, q) = Rkq;
		}
	}


	template<typename Real>
	COMM_FUNC void EigenDecomposition(const SquareMatrix<Real, 3> &A, SquareMatrix<Real, 3> &eigenVecs, Vector<Real, 3> &eigenVals)
	{
		const int numJacobiIterations = 10;
		const Real epsilon = 1e-15f;

		SquareMatrix<Real, 3> D = A;

		// only for symmetric matrices!
		eigenVecs = SquareMatrix<Real, 3>::identityMatrix();	// unit matrix
		int iter = 0;
		while (iter < numJacobiIterations) {	// 3 off diagonal elements
												// find off diagonal element with maximum modulus
			int p, q;
			Real a, max;
			max = fabs(D(0, 1));
			p = 0; q = 1;
			a = fabs(D(0, 2));
			if (a > max) { p = 0; q = 2; max = a; }
			a = fabs(D(1, 2));
			if (a > max) { p = 1; q = 2; max = a; }
			// all small enough -> done
			if (max < epsilon) break;
			// rotate matrix with respect to that element
			jacobiRotate<Real>(D, eigenVecs, p, q);
			iter++;
		}
		eigenVals[0] = D(0, 0);
		eigenVals[1] = D(1, 1);
		eigenVals[2] = D(2, 2);
	}

	template<typename Real>
	COMM_FUNC void polarDecomposition(const SquareMatrix<Real, 3> &A, SquareMatrix<Real, 3> &R, SquareMatrix<Real, 3> &U, SquareMatrix<Real, 3> &D)
	{
		// A = SR, where S is symmetric and R is orthonormal
		// -> S = (A A^T)^(1/2)

		// A = U D U^T R

		SquareMatrix<Real, 3> AAT;
		AAT(0, 0) = A(0, 0)*A(0, 0) + A(0, 1)*A(0, 1) + A(0, 2)*A(0, 2);
		AAT(1, 1) = A(1, 0)*A(1, 0) + A(1, 1)*A(1, 1) + A(1, 2)*A(1, 2);
		AAT(2, 2) = A(2, 0)*A(2, 0) + A(2, 1)*A(2, 1) + A(2, 2)*A(2, 2);

		AAT(0, 1) = A(0, 0)*A(1, 0) + A(0, 1)*A(1, 1) + A(0, 2)*A(1, 2);
		AAT(0, 2) = A(0, 0)*A(2, 0) + A(0, 1)*A(2, 1) + A(0, 2)*A(2, 2);
		AAT(1, 2) = A(1, 0)*A(2, 0) + A(1, 1)*A(2, 1) + A(1, 2)*A(2, 2);

		AAT(1, 0) = AAT(0, 1);
		AAT(2, 0) = AAT(0, 2);
		AAT(2, 1) = AAT(1, 2);

		R = SquareMatrix<Real, 3>::identityMatrix();
		Vector<Real, 3> eigenVals;
		EigenDecomposition<Real>(AAT, U, eigenVals);

		Real d0 = sqrt(eigenVals[0]);
		Real d1 = sqrt(eigenVals[1]);
		Real d2 = sqrt(eigenVals[2]);
		D = SquareMatrix<Real, 3>(0);
		D(0, 0) = d0;
		D(1, 1) = d1;
		D(2, 2) = d2;

		const Real eps = 1e-15f;

		Real l0 = eigenVals[0]; if (l0 <= eps) l0 = 0.0f; else l0 = 1.0f / d0;
		Real l1 = eigenVals[1]; if (l1 <= eps) l1 = 0.0f; else l1 = 1.0f / d1;
		Real l2 = eigenVals[2]; if (l2 <= eps) l2 = 0.0f; else l2 = 1.0f / d2;

		SquareMatrix<Real, 3> S1;
		S1(0, 0) = l0*U(0, 0)*U(0, 0) + l1*U(0, 1)*U(0, 1) + l2*U(0, 2)*U(0, 2);
		S1(1, 1) = l0*U(1, 0)*U(1, 0) + l1*U(1, 1)*U(1, 1) + l2*U(1, 2)*U(1, 2);
		S1(2, 2) = l0*U(2, 0)*U(2, 0) + l1*U(2, 1)*U(2, 1) + l2*U(2, 2)*U(2, 2);

		S1(0, 1) = l0*U(0, 0)*U(1, 0) + l1*U(0, 1)*U(1, 1) + l2*U(0, 2)*U(1, 2);
		S1(0, 2) = l0*U(0, 0)*U(2, 0) + l1*U(0, 1)*U(2, 1) + l2*U(0, 2)*U(2, 2);
		S1(1, 2) = l0*U(1, 0)*U(2, 0) + l1*U(1, 1)*U(2, 1) + l2*U(1, 2)*U(2, 2);

		S1(1, 0) = S1(0, 1);
		S1(2, 0) = S1(0, 2);
		S1(2, 1) = S1(1, 2);

		R = S1*A;

		// stabilize
		Vector<Real, 3> c0, c1, c2;
		// 		c0 = R.col(0);
		// 		c1 = R.col(1);
		// 		c2 = R.col(2);
		c0[0] = R(0, 0);	c1[0] = R(0, 1);	c2[0] = R(0, 2);
		c0[1] = R(1, 0);	c1[1] = R(1, 1);	c2[1] = R(1, 2);
		c0[2] = R(2, 0);	c1[2] = R(2, 1);	c2[2] = R(2, 2);

		if (c0.normSquared() < eps)
			c0 = c1.cross(c2);
		else if (c1.normSquared() < eps)
			c1 = c2.cross(c0);
		else
			c2 = c0.cross(c1);
		// 		R.col(0) = c0;
		// 		R.col(1) = c1;
		// 		R.col(2) = c2;
		R(0, 0) = c0[0];	R(0, 1) = c1[0];	R(0, 2) = c2[0];
		R(1, 0) = c0[1];	R(1, 1) = c1[1];	R(1, 2) = c2[1];
		R(2, 0) = c0[2];	R(2, 1) = c1[2];	R(2, 2) = c2[2];
	}

	template <typename Real>
	COMM_FUNC SquareMatrix<Real, 3> inverse(const SquareMatrix<Real, 3> mat)
	{

	}


	template <typename Scalar>
	COMM_FUNC SquareMatrix<Real, 2> inverse(const SquareMatrix<Real, 2> mat)
	{

	}
}