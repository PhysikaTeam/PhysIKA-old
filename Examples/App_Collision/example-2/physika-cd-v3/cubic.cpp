#define NEW_SOLVER
#ifdef NEW_SOLVER
#include <math.h>

// Solving cubic equations
int solve_cubic (double a, double b, double c, double d, double x[3]);

double newtons_method (double a, double b, double c, double d, double x0,
                       int init_dir);

int solveCubic(double c[4], double s[3])
{
	return solve_cubic(c[3], c[2], c[1], c[0], s);
}

template <typename T> T sgn (const T &x) {return x<0 ? -1 : 1;}

inline void swap(double &a, double &b) {
	double c = a;
	a = b;
	b = c;
}

int solve_quadratic (double a, double b, double c, double x[2]) {
    // http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
    double d = b*b - 4*a*c;
    if (d < 0) {
        x[0] = -b/(2*a);
        return 0;
    }
    double q = -(b + sgn(b)*sqrt(d))/2;
    int i = 0;
    if (fabs(a) > 1e-12*fabs(q))
        x[i++] = q/a;
    if (fabs(q) > 1e-12*fabs(c))
        x[i++] = c/q;
    if (i==2 && x[0] > x[1])
        swap(x[0], x[1]);
    return i;
}


// solves a x^3 + b x^2 + c x + d == 0
int solve_cubic (double a, double b, double c, double d, double x[3]) {
    double xc[2];
    int ncrit = solve_quadratic(3*a, 2*b, c, xc);
    if (ncrit == 0) {
        x[0] = newtons_method(a, b, c, d, xc[0], 0);
        return 1;
    } else if (ncrit == 1) {// cubic is actually quadratic
        return solve_quadratic(b, c, d, x);
    } else {
        double yc[2] = {d + xc[0]*(c + xc[0]*(b + xc[0]*a)),
                        d + xc[1]*(c + xc[1]*(b + xc[1]*a))};
        int i = 0;
        if (yc[0]*a >= 0)
            x[i++] = newtons_method(a, b, c, d, xc[0], -1);
        if (yc[0]*yc[1] <= 0) {
            int closer = fabs(yc[0])<fabs(yc[1]) ? 0 : 1;
            x[i++] = newtons_method(a, b, c, d, xc[closer], closer==0?1:-1);
        }
        if (yc[1]*a <= 0)
            x[i++] = newtons_method(a, b, c, d, xc[1], 1);
        return i;
    }
}

double newtons_method (double a, double b, double c, double d, double x0,
                       int init_dir) {
    if (init_dir != 0) {
        // quadratic approximation around x0, assuming y' = 0
        double y0 = d + x0*(c + x0*(b + x0*a)),
               ddy0 = 2*b + x0*(6*a);
        x0 += init_dir*sqrt(fabs(2*y0/ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        double y = d + x0*(c + x0*(b + x0*a));
        double dy = c + x0*(2*b + x0*3*a);
        if (dy == 0)
            return x0;
        double x1 = x0 - y/dy;
        if (fabs(x0 - x1) < 1e-6)
            return x0;
        x0 = x1;
    }
    return x0;
}

#else

/****************************************************************
*								*
* Polynomial root finder (polynoms up to degree 4)		*
* AUTHOR: Jochen SCHARZE (See 'Cubic & Quartic Roots' in	*
*			  'Graphics Gems 1', AP)		*
*								*
****************************************************************/

#include <math.h>
#define EQN_EPS double(1e-7)
#ifndef M_PI
#define M_PI        3.14159265358979323846f
#endif

#define ONE_DIV_3  0.33333333333333333f

int isZero(double x);
int solveCubic(double c[4], double s[3]);

double cbrt(double arg) {
	return pow(arg, 1.0/3.0);
}

/********************************************************
*							*
* This function determines if a double is small enough	*
* to be zero. The purpose of the subroutine is to try	*
* to overcome precision problems in math routines.	*
*							*
********************************************************/

static int isZero(double x)
{
	return x > -EQN_EPS && x < EQN_EPS;
}

int solveLinear(double c[2], double s[1])
{
	if (isZero(c[1]))
		return 0;
	s[0] = - c[0] / c[1];
	return 1;
}



/********************************************************
*							*
* This function determines the roots of a quadric	*
* equation.						*
* It takes as parameters a pointer to the three		*
* coefficient of the quadric equation (the c[2] is the	*
* coefficient of x2 and so on) and a pointer to the	*
* two element array in which the roots are to be	*
* placed.						*
* It outputs the number of roots found.			*
*							*
********************************************************/

int solveQuadric(double c[3], double s[2])
{
	double p, q, D;

	// make sure we have a d2 equation

	if (isZero(c[2]))
		return solveLinear(c, s);

	// normal for: x^2 + px + q
	p = c[1] / (2.0f * c[2]);
	q = c[0] / c[2];
	D = p * p - q;

	if (isZero(D))
	{
		// one double root
		s[0] = s[1] = -p;
		return 1;
	}

	if (D < 0.0)
		// no real root
		return 0;

	else
	{
		// two real roots
		double sqrt_D = sqrt(D);
		s[0] = sqrt_D - p;
		s[1] = -sqrt_D - p;
		return 2;
	}
}


/********************************************************
*							*
* This function determines the roots of a cubic		*
* equation.						*
* It takes as parameters a pointer to the four		*
* coefficient of the cubic equation (the c[3] is the	*
* coefficient of x3 and so on) and a pointer to the	*
* three element array in which the roots are to be	*
* placed.						*
* It outputs the number of roots found			*
*							*
********************************************************/

int solveCubic(double c[4], double s[3])
{
	int	i, num;
	double	sub,
		A, B, C,
		sq_A, p, q,
		cb_p, D;

	// make sure we have a d2 equation

	if (isZero(c[3]))
		return solveQuadric(c, s);

	// normalize the equation:x ^ 3 + Ax ^ 2 + Bx  + C = 0
	A = c[2] / c[3];
	B = c[1] / c[3];
	C = c[0] / c[3];

	// substitute x = y - A / 3 to eliminate the quadric term: x^3 + px + q = 0

	sq_A = A * A;
	p = ONE_DIV_3 * (-ONE_DIV_3 * sq_A + B);
	q = 0.5f * (2.0f/27.0f * A *sq_A - ONE_DIV_3 * A * B + C);

	// use Cardano's formula

	cb_p = p * p * p;
	D = q * q + cb_p;

	if (isZero(D))
	{
		if (isZero(q))
		{
			// one triple s
			s[0] = 0.0;
			num = 1;
		}
		else
		{
			// one single and one double s
			double u = cbrt(-q);
			s[0] = 2.0f * u;
			s[1] = - u;
			num = 2;
		}
	}
	else
		if (D < 0.0)
		{
			// casus irreductibilis: three real solutions
			double phi = ONE_DIV_3 * acos(-q / sqrt(-cb_p));
			double t = 2.0f * sqrt(-p);
			s[0] = t * cos(phi);
			s[1] = -t * cos(phi + M_PI / 3.0f);
			s[2] = -t * cos(phi - M_PI / 3.0f);
			num = 3;
		}
		else
		{
			// one real s
			double sqrt_D = sqrt(D);
			double u = cbrt(sqrt_D + fabs(q));
			if (q > 0.0)
				s[0] = - u + p / u ;
			else
				s[0] = u - p / u;
			num = 1;
		}

		// resubstitute
		sub = ONE_DIV_3 * A;
		for (i = 0; i < num; i++)
			s[i] -= sub;

		//sort
		if (num == 2 && s[1] < s[0]) {
			double t=s[1];
			s[1] = s[0];
			s[0] = t;
		}

		if (num == 3) {
			// Bubblesort
			if ( s[0] > s[1] ) {
				double tmp = s[0]; s[0] = s[1]; s[1] = tmp;
			}
			if ( s[1] > s[2] ) {
				double tmp = s[1]; s[1] = s[2]; s[2] = tmp;
			}
			if ( s[0] > s[1] ) {
				double tmp = s[0]; s[0] = s[1]; s[1] = tmp;
			}
		}

		return num;
}
#endif
