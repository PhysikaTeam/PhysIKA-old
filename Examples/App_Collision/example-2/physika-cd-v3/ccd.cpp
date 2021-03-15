#include "vec3f.h"
#include "cubic.h"
#include <stdio.h>

//#define ccdTimeResolution GLH_EPSILON
#define     ccdTimeResolution         double(10e-8)

#define zeroRes double(10e-11)
#define IsZero(x) ((x) < zeroRes && (x) > -zeroRes ? true : false)

typedef struct {
	vec3f ad, bd, cd, pd;
	vec3f a0, b0, c0, p0;
} NewtonCheckData;

/*
* Ordinary inside-triangle test for p. The triangle normal is computed from the vertices.
*/
static inline bool _insideTriangle(const vec3f &a, const vec3f &b, const vec3f &c, const vec3f &p, vec3f &baryc)
{
	vec3f n, da, db, dc;
	double wa, wb, wc;

	vec3f ba = b-a;
	vec3f ca = c-a;
	n = ba.cross(ca);

	da = a - p, db = b - p, dc = c - p;
	if ((wa = (db.cross(dc)).dot(n)) < 0.0f) return false;
	if ((wb = (dc.cross(da)).dot(n)) < 0.0f) return false;
	if ((wc = (da.cross(db)).dot(n)) < 0.0f) return false;

	//Compute barycentric coordinates
	double area2 = n.dot(n);
	wa /= area2, wb /= area2, wc /= area2;

	baryc = vec3f(wa, wb, wc);

	return true;
}

static inline bool _insideLnSeg(const vec3f &a, const vec3f &b, const vec3f &c)
{
	return (a-b).dot(a-c)<=0;
}

static inline bool checkRootValidity_VE(double t, NewtonCheckData &data)
{
	return _insideLnSeg(
		data.ad*t + data.a0, 
		data.bd*t + data.b0, 
		data.cd*t + data.c0);

}

static inline bool checkRootValidity_VF(double t, vec3f& baryc, NewtonCheckData &data) {
	return _insideTriangle(
		data.ad*t + data.a0, 
		data.bd*t + data.b0, 
		data.cd*t + data.c0, 
		data.pd*t + data.p0,
		baryc);
}


// http://local.wasp.uwa.edu.au/~pbourke/geometry/lineline3d/

/*
Calculate the line segment PaPb that is the shortest route between
two lines P1P2 and P3P4. Calculate also the values of mua and mub where
Pa = P1 + mua (P2 - P1)
Pb = P3 + mub (P4 - P3)
Return FALSE if no solution exists.
*/
inline bool LineLineIntersect(
							  const vec3f &p1, const vec3f &p2, const vec3f &p3, const vec3f &p4,
							  vec3f &pa, vec3f &pb, double &mua, double &mub)
{
	vec3f p13,p43,p21;
	double d1343,d4321,d1321,d4343,d2121;
	double numer,denom;

	p13 = p1 - p3;
	p43 = p4 - p3;
	if (fabs(p43[0])  < GLH_EPSILON && fabs(p43[1])  < GLH_EPSILON && fabs(p43[2])  < GLH_EPSILON)
		return false;

	p21 = p2 - p1;
	if (fabs(p21[0])  < GLH_EPSILON && fabs(p21[1])  < GLH_EPSILON && fabs(p21[2])  < GLH_EPSILON)
		return false;

	d1343 = p13.dot(p43);
	d4321 = p43.dot(p21);
	d1321 = p13.dot(p21);
	d4343 = p43.dot(p43);
	d2121 = p21.dot(p21);

	denom = d2121 * d4343 - d4321 * d4321;
	if (fabs(denom) < GLH_EPSILON_2)
		return false;
	numer = d1343 * d4321 - d1321 * d4343;

	mua = numer / denom;
	if (mua < 0 || mua > 1)
		return false;

	mub = (d1343 + d4321 * mua) / d4343;
	if (mub < 0 || mub > 1)
		return false;

	pa = p1 + p21*mua;
	pb = p3 + p43*mub;
	return true;
}

static inline bool checkRootValidity_EE(double t, vec3f &pt, NewtonCheckData &data) {
	vec3f a = data.ad*t + data.a0;
	vec3f b = data.bd*t + data.b0;
	vec3f c = data.cd*t + data.c0;
	vec3f d = data.pd*t + data.p0;

	vec3f p1, p2;
	double tab, tcd;

	if (LineLineIntersect(a, b, c, d, p1, p2, tab, tcd)) {
		t = tab;
		pt = p1;
		return true;
	}

	return false;
}

/*
* Computes the coefficients of the cubic equations from the geometry data.
*/
inline void _equateCubic_VF(const vec3f &a0, const vec3f &ad, const vec3f &b0, const vec3f &bd, 
							const vec3f &c0, const vec3f &cd, const vec3f &p0, const vec3f &pd,
							double &a, double &b, double &c, double &d)
{
	/*
	* For definitions & notation refer to the semester thesis doc.
	*/
	vec3f dab, dac, dap;
	vec3f oab, oac, oap;
	vec3f dabXdac, dabXoac, oabXdac, oabXoac;

	dab = bd - ad, dac = cd - ad, dap = pd - ad;
	oab = b0 - a0, oac = c0 - a0, oap = p0 - a0;
	dabXdac = dab.cross(dac);
	dabXoac = dab.cross(oac);
	oabXdac = oab.cross(dac);
	oabXoac = oab.cross(oac);

	a = dap.dot(dabXdac);
	b = oap.dot(dabXdac) + dap.dot(dabXoac + oabXdac);
	c = dap.dot(oabXoac) + oap.dot(dabXoac + oabXdac);
	d = oap.dot(oabXoac);
}

/*
* Computes the coefficients of the cubic equations from the geometry data.
*/
inline void _equateCubic_VE(
							const vec3f &a0, const vec3f &ad, const vec3f &b0, const vec3f &bd, 
							const vec3f &c0, const vec3f &cd, const vec3f &L,
							double &a, double &b, double &c)
{
	/*
	* For definitions & notation refer to the semester thesis doc.
	*/
	vec3f dab, dcb;
	vec3f oab, ocb;

	dab = ad-bd; dcb = cd-bd;
	oab = a0-b0; ocb = c0-b0;

	vec3f Ldcb = L.cross(dcb);
	vec3f Locb = L.cross(ocb);

	a = Ldcb.dot(dab);
	b = Ldcb.dot(oab) + Locb.dot(dab);
	c = Locb.dot(oab);
}

/*
* Computes the coefficients of the cubic equations from the geometry data.
*/
inline void _equateCubic_EE(const vec3f &a0, const vec3f &ad, const vec3f &b0, const vec3f &bd, 
							const vec3f &c0, const vec3f &cd, const vec3f &d0, const vec3f &dd,
							double &a, double &b, double &c, double &d)
{
	/*
	* For definitions & notation refer to the semester thesis doc.
	*/
	vec3f dba, ddc, dca;
	vec3f odc, oba, oca;
	vec3f dbaXddc, dbaXodc, obaXddc, obaXodc;

	dba = bd - ad, ddc = dd - cd, dca = cd - ad;
	odc = d0 - c0, oba = b0 - a0, oca = c0 - a0;
	dbaXddc = dba.cross(ddc);
	dbaXodc = dba.cross(odc);
	obaXddc = oba.cross(ddc);
	obaXodc = oba.cross(odc);

	a = dca.dot(dbaXddc);
	b = oca.dot(dbaXddc) + dca.dot(dbaXodc + obaXddc);
	c = dca.dot(obaXodc) + oca.dot(dbaXodc + obaXddc);
	d = oca.dot(obaXodc);
}

double
Intersect_VF(const vec3f &ta0, const vec3f &tb0, const vec3f &tc0,
			 const vec3f &ta1, const vec3f &tb1, const vec3f &tc1,
			 const vec3f &q0, const vec3f &q1,
			 vec3f &qi, vec3f &baryc)
{
	/* Default value returned if no collision occurs */
	double collisionTime = -1.0f;

	vec3f qd, ad, bd, cd;
	/* diff. vectors for linear interpolation */
	qd = q1 - q0, ad = ta1 - ta0, bd = tb1 - tb0, cd = tc1 - tc0;

	/*
	* Compute scalar coefficients by evaluating dot and cross-products.
	*/
	double a, b, c, d; /* cubic polynomial coefficients */
	_equateCubic_VF(ta0, ad, tb0, bd, tc0, cd, q0, qd, a, b, c, d);

	if (IsZero(a) && IsZero(b) && IsZero(c) && IsZero(d))
		return -1.f;

	double roots[3];
	double coeffs[4];
	coeffs[3] = a, coeffs[2] = b, coeffs[1] = c, coeffs[0] = d;
	int num = solveCubic(coeffs, roots);

	if (num == 0)
		return -1.f;

	NewtonCheckData data;
	data.a0 = ta0, data.b0 = tb0;
	data.c0 = tc0, data.p0 = q0;
	data.ad = ad, data.bd = bd;
	data.cd = cd, data.pd = qd;

	for (int i=0; i<num; i++) {
		double r = roots[i];
		if (r < 0 || r > 1) continue;

		if (checkRootValidity_VF(r, baryc, data)) {
			collisionTime = r;
			break;
		}
	}

	if (collisionTime >= 0)
		qi = qd*collisionTime + q0;				

	return collisionTime;
}


///////////////////////////////////////////////////////////////////////////////////////////////
const double flat_tol =1e-6;
const double one_div_tree = 1./3.;

class bezier_info {
public:
	double a, b, c, d;
	double t0, t1;

public:
	bezier_info() {
		a = b = c = d = t0 = t1 = -1.;
	}

	void set(double ta, double tb, double tc, double td, double tt0, double tt1) {
		a = ta, b = tb, c = tc, d = td;
		t0 = tt0, t1 = tt1;
	}

	bool isFlat() {
		return	(t1-t0)<flat_tol;

/*		double m1 = (a+d)*one_div_tree;
		double m2 = m1*2.;

		return (fabs(b-m1) < flat_tol && fabs(c-m2) < flat_tol);*/
	}

	double linearRoot() {
		return t1;

//		return (t0-t1)*a/(d-a);
	}

	void split(bezier_info &c1, bezier_info &c2) {
		double e = (a+b)*0.5;
		double f = (b+c)*0.5;
		double g = (c+d)*0.5;
		double h = (e+f)*0.5;
		double i=(f+g)*0.5;
		double j=(h+i)*0.5;

		double tt = (t0+t1)*0.5;
		c1.set(a, e, h, j, t0, tt);
		c2.set(j, i, g, d, tt, t1);
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
#define STACK_DEPTH 25
#define EMPTY(idx) (idx == 0)

#define PUSH_PAIR(crv, stack, idx)  {\
	stack[idx++] = crv;\
	\
	if (idx > STACK_DEPTH) {\
		printf("bezier stack overflow ...\n");\
	}\
}

#define POP_PAIR(crv, stack, idx) {\
	if (idx == 0) {\
		;\
	} else {\
		idx--;\
		crv = stack[idx];\
	}\
}

#define NEXT(crv, stack, idx) {\
	POP_PAIR(crv, stack, idx)\
}

bool findRoots(bezier_info crv, double *roots, int &num)
{
	bezier_info bstack[STACK_DEPTH];
	int idx = 0;

	num = 0;
	PUSH_PAIR(crv, bstack, idx);
	while (!EMPTY(idx)) {
		NEXT(crv, bstack, idx);

		if ((crv.a > 0 && crv.d < 0) || (crv.a < 0 && crv.d > 0)) { // must have > 1 root
			if (crv.isFlat()) {
				roots[num++] = crv.linearRoot();
				if (num == 4) return true;
				continue;
			}
		}

		if ((crv.a > 0 && crv.d > 0 && crv.c > 0 && crv.b > 0) ||
			(crv.a < 0 && crv.d < 0 && crv.c < 0 && crv.b < 0))
			continue; // no root for this segement

		bezier_info c1, c2;
		crv.split(c1, c2);
		PUSH_PAIR(c1, bstack, idx);
		PUSH_PAIR(c2, bstack, idx);
	}

	// sorting the roots
	if (num == 2) {
		if (roots[0] > roots[1]) {
			double t = roots[0];
			roots[0] = roots[1];
			roots[1] = t;
		}
	}
	if (num == 3) {
		// Bubblesort
		if ( roots[0] > roots[1] ) {
			double tmp = roots[0]; roots[0] = roots[1]; roots[1] = tmp;
		}
		if ( roots[1] > roots[2] ) {
			double tmp = roots[1]; roots[1] = roots[2]; roots[2] = tmp;
		}
		if ( roots[0] > roots[1] ) {
			double tmp = roots[0]; roots[0] = roots[1]; roots[1] = tmp;
		}
	}

	return num > 0;
}

inline vec3f norm(const vec3f &p1, const vec3f &p2, const vec3f &p3)
{
	return (p2-p1).cross(p3-p1);
}

void find_coplanarity_times(
			const vec3f &ta0, const vec3f &tb0, const vec3f &tc0, const vec3f &td0,
			const vec3f &ta1, const vec3f &tb1, const vec3f &tc1, const vec3f &td1,
            double *roots, int &num)
  {
	vec3f n0 = norm(ta0, tb0, tc0);
	vec3f n1 = norm(ta1, tb1, tc1);
	vec3f delta = norm(ta1-ta0, tb1-tb0, tc1-tc0);
	vec3f nX = (n0+n1-delta)*0.5;

	vec3f pa0 = td0-ta0;
	vec3f pa1 = td1-ta1;

	double A = n0.dot(pa0);
	double B = n1.dot(pa1);
	double C = nX.dot(pa0);
	double D = nX.dot(pa1);
	double E = n1.dot(pa0);
	double F = n0.dot(pa1);

	double X = 2*C+F;
	double Y = 2*D+E;

	num = 0;
	if (IsZero(A) && IsZero(B) && IsZero(X) && IsZero(Y))
		return;

	if (A > 0 && B > 0 && X > 0 && Y > 0)
		return;

	if (A < 0 && B < 0 && X < 0 && Y < 0)
		return;

	bezier_info  crv;
	crv.set(A, X, Y, B, 0, 1);

	findRoots(crv, roots, num);
 }

extern "C" void find_coplanarity_times_export(
			const void *ta0, const void *tb0, const void *tc0, const void *td0,
			const void *ta1, const void *tb1, const void *tc1, const void *td1,
            double *roots, int &num)
{
	vec3f a0 = *((vec3f *)ta0);
	vec3f b0 = *((vec3f *)tb0);
	vec3f c0 = *((vec3f *)tc0);
	vec3f d0 = *((vec3f *)td0);
	vec3f a1 = *((vec3f *)ta1);
	vec3f b1 = *((vec3f *)tb1);
	vec3f c1 = *((vec3f *)tc1);
	vec3f d1 = *((vec3f *)td1);

	find_coplanarity_times(a0, b0, c0, d0, a1, b1, c1, d1, roots, num);
}

double
Intersect_VF2(const vec3f &ta0, const vec3f &tb0, const vec3f &tc0,
			 const vec3f &ta1, const vec3f &tb1, const vec3f &tc1,
			 const vec3f &q0, const vec3f &q1,
			 vec3f &qi, vec3f &baryc)
{
	double roots[3];
	int num=0;
	find_coplanarity_times(ta0, tb0, tc0, q0, ta1, tb1, tc1, q1, roots, num);

	if (num == 0)
		return -1.0;

	vec3f qd, ad, bd, cd;
	/* diff. vectors for linear interpolation */
	qd = q1 - q0, ad = ta1 - ta0, bd = tb1 - tb0, cd = tc1 - tc0;
	double collisionTime = -1.0f;

	NewtonCheckData data;
	data.a0 = ta0, data.b0 = tb0;
	data.c0 = tc0, data.p0 = q0;
	data.ad = ad, data.bd = bd;
	data.cd = cd, data.pd = qd;

	for (int i=0; i<num; i++) {
		double r = roots[i];
		if (r < 0 || r > 1) continue;

		if (checkRootValidity_VF(r, baryc, data)) {
			collisionTime = r;
			break;
		}
	}

	if (collisionTime >= 0)
		qi = qd*collisionTime + q0;				

	return collisionTime;
}

double
Intersect_EE1(const vec3f &ta0, const vec3f &tb0, const vec3f &tc0, const vec3f &td0,
			 const vec3f &ta1, const vec3f &tb1, const vec3f &tc1, const vec3f &td1,
			 vec3f &qi)
{
	/* Default value returned if no collision occurs */
	double collisionTime = -1.0f;

	vec3f ad, bd, cd, dd;
	/* diff. vectors for linear interpolation */
	dd = td1 - td0, ad = ta1 - ta0, bd = tb1 - tb0, cd = tc1 - tc0;

	/*
	* Compute scalar coefficients by evaluating dot and cross-products.
	*/
	double a, b, c, d; /* cubic polynomial coefficients */
	_equateCubic_EE(ta0, ad, tb0, bd, tc0, cd, td0, dd, a, b, c, d);

	if (IsZero(a) && IsZero(b) && IsZero(c) && IsZero(d))
		return -1.f;

	double roots[3];
	double coeffs[4];
	coeffs[3] = a, coeffs[2] = b, coeffs[1] = c, coeffs[0] = d;
	int num = solveCubic(coeffs, roots);

	if (num == 0)
		return -1.f;

	NewtonCheckData data;
	data.a0 = ta0, data.b0 = tb0;
	data.c0 = tc0, data.p0 = td0;
	data.ad = ad, data.bd = bd;
	data.cd = cd, data.pd = dd;

	for (int i=0; i<num; i++) {
		double r = roots[i];
		if (r < 0 || r > 1) continue;

		if (checkRootValidity_EE(r, qi, data)) {
			collisionTime = r;
			break;
		}
	}

	return collisionTime;
}

double
Intersect_EE(const vec3f &ta0, const vec3f &tb0, const vec3f &tc0, const vec3f &td0,
			 const vec3f &ta1, const vec3f &tb1, const vec3f &tc1, const vec3f &td1,
			 vec3f &qi)
{
	double t = Intersect_EE1(ta0, tb0, tc0, td0, ta1, tb1, tc1, td1, qi);

	double roots[3];
	int num=0;
	find_coplanarity_times(ta0, tb0, tc0, td0, ta1, tb1, tc1, td1, roots, num);

	if (num == 0)
		return -1.0;

	vec3f ad, bd, cd, dd;
	/* diff. vectors for linear interpolation */
	dd = td1 - td0, ad = ta1 - ta0, bd = tb1 - tb0, cd = tc1 - tc0;

	NewtonCheckData data;
	data.a0 = ta0, data.b0 = tb0;
	data.c0 = tc0, data.p0 = td0;
	data.ad = ad, data.bd = bd;
	data.cd = cd, data.pd = dd;

	/* Default value returned if no collision occurs */
	double collisionTime = -1.0f;

	for (int i=0; i<num; i++) {
		double r = roots[i];
		if (r < 0 || r > 1) continue;

		if (checkRootValidity_EE(r, qi, data)) {
			collisionTime = r;
			break;
		}
	}

	t = Intersect_EE1(ta0, tb0, tc0, td0, ta1, tb1, tc1, td1, qi);
	return collisionTime;
}