#pragma once

// very robust triangle intersection test
// uses no divisions
// works on coplanar triangles

inline __device__ int project3(const REAL3 &ax,
	const REAL3 &p1, const REAL3 &p2, const REAL3 &p3)
{
	REAL P1 = dot(ax, p1);
	REAL P2 = dot(ax, p2);
	REAL P3 = dot(ax, p3);

	REAL mx1 = fmaxf(P1, fmaxf(P2, P3));
	REAL mn1 = fminf(P1, fminf(P2, P3));

	if (mn1 > 0) return 0;
	if (0 > mx1) return 0;

	return 1;
}

inline  __device__ int project6(REAL3 &ax,
	REAL3 &p1, REAL3 &p2, REAL3 &p3,
	REAL3 &q1, REAL3 &q2, REAL3 &q3)
{
	REAL P1 = dot(ax, p1);
	REAL P2 = dot(ax, p2);
	REAL P3 = dot(ax, p3);
	REAL Q1 = dot(ax, q1);
	REAL Q2 = dot(ax, q2);
	REAL Q3 = dot(ax, q3);

	REAL mx1 = fmaxf(P1, fmaxf(P2, P3));
	REAL mn1 = fminf(P1, fminf(P2, P3));
	REAL mx2 = fmaxf(Q1, fmaxf(Q2, Q3));
	REAL mn2 = fminf(Q1, fminf(Q2, Q3));

	if (mn1 > mx2) return 0;
	if (mn2 > mx1) return 0;

	return 1;
}

inline __device__ bool
tri_contact(REAL3 &P1, REAL3 &P2, REAL3 &P3,
	REAL3 &Q1, REAL3 &Q2, REAL3 &Q3)
{
	REAL3 p1 = zero3f();;
	REAL3 p2 = P2 - P1;
	REAL3 p3 = P3 - P1;
	REAL3 q1 = Q1 - P1;
	REAL3 q2 = Q2 - P1;
	REAL3 q3 = Q3 - P1;

	REAL3 e1 = p2 - p1;
	REAL3 e2 = p3 - p2;
	REAL3 e3 = p1 - p3;

	REAL3 f1 = q2 - q1;
	REAL3 f2 = q3 - q2;
	REAL3 f3 = q1 - q3;

	REAL3 n1 = cross(e1, e2);
	REAL3 m1 = cross(f1, f2);

	REAL3 g1 = cross(e1, n1);
	REAL3 g2 = cross(e2, n1);
	REAL3 g3 = cross(e3, n1);

	REAL3 h1 = cross(f1, m1);
	REAL3 h2 = cross(f2, m1);
	REAL3 h3 = cross(f3, m1);

	REAL3 ef11 = cross(e1, f1);
	REAL3 ef12 = cross(e1, f2);
	REAL3 ef13 = cross(e1, f3);
	REAL3 ef21 = cross(e2, f1);
	REAL3 ef22 = cross(e2, f2);
	REAL3 ef23 = cross(e2, f3);
	REAL3 ef31 = cross(e3, f1);
	REAL3 ef32 = cross(e3, f2);
	REAL3 ef33 = cross(e3, f3);

	// now begin the series of tests
	if (!project3(n1, q1, q2, q3)) return false;
	if (!project3(m1, -q1, p2 - q1, p3 - q1)) return false;

	if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

	return true;
}