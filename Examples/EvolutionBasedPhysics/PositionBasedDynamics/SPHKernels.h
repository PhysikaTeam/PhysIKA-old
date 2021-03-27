#ifndef SPHKERNELS_H
#define SPHKERNELS_H

#define _USE_MATH_DEFINES
#include <math.h>
#include "Common/Common.h"
#include <algorithm>

#define NO_DISTANCE_TEST

namespace PBD
{
	class CubicKernel
	{
	protected:
		static Real m_radius;
		static Real m_k;
		static Real m_l;
		static Real m_W_zero;
	public:
		static Real getRadius() { return m_radius; }
		static void setRadius(Real val)
		{
			m_radius = val;
			static const Real pi = static_cast<Real>(M_PI);

			const Real h3 = m_radius*m_radius*m_radius;
			m_k = 8.0 / (pi*h3);
			m_l = 48.0 / (pi*h3);
			m_W_zero = W(Vector3r(0.0, 0.0, 0.0));
		}

	public:
		//static unsigned int counter;
		static Real W(const Vector3r &r)
		{
			//counter++;
			Real res = 0.0;
			const Real rl = r.norm();
			const Real q = rl/m_radius;
#ifndef NO_DISTANCE_TEST
			if (q <= 1.0)
#endif
			{
				if (q <= 0.5)
				{
					const Real q2 = q*q;
					const Real q3 = q2*q;
					res = m_k * (6.0*q3-6.0*q2+1.0);
				}
				else
				{
					res = m_k * (2.0*pow(1.0-q,3));
				}
			}
			return res;
		}

		static Vector3r gradW(const Vector3r &r)
		{
			Vector3r res;
			const Real rl = r.norm();
			const Real q = rl / m_radius;
#ifndef NO_DISTANCE_TEST
			if (q <= 1.0)
#endif
			{
				if (rl > 1.0e-6)
				{
					const Vector3r gradq = r * ((Real) 1.0 / (rl*m_radius));
					if (q <= 0.5)
					{
						res = m_l*q*((Real) 3.0*q - (Real) 2.0)*gradq;
					}
					else
					{
						const Real factor = 1.0 - q;
						res = m_l*(-factor*factor)*gradq;
					}
				}
			}
#ifndef NO_DISTANCE_TEST
 			else
 				res.zero();
#endif

			return res;
		}

		static Real W_zero()
		{
			return m_W_zero;
		}
		
		// 论文“Particle-Based Fluid Simulation for Interactive Applications”用的核函数
		// Poly6核函数
		static Real W_poly6(const Vector3r &r)
		{
			Real res;
			Real m_k;
			const Real rl = r.norm();
			Real h9 = pow(m_radius, 9);
			m_k = 315 / (64 * M_PI*h9);
			if (rl >= 0 && rl <= m_radius)
			{
				Real h2 = m_radius*m_radius;
				Real r2 = rl*rl;
				Real hr = h2 - r2;
				res = m_k*hr*hr*hr;
			}
			else
			{
				res = 0;
			}
			return res;
		}
		//Poly6核函数的梯度
		static Vector3r grad_W_poly6(const Vector3r &r)
		{
			Vector3r res;
			Real h9 = pow(m_radius, 9);
			Real m_k;
			m_k = 945.0 / (32 * M_PI * h9);
			Real rl = r.norm();
			if (rl >= 0 && rl <= m_radius)
			{
				Real h2 = m_radius*m_radius;
				Real r2 = rl*rl;
				Real hr = h2 - r2;
				res = -1 * m_k*hr*hr*r;
			}
			else
			{
				res.setZero();
			}
			return res;
		}
		//Poly6核函数的Laplacian
		static Real laplacian_W_poly6(const Vector3r &r)
		{
			Real res;
			Real h9 = pow(m_radius, 9);
			Real m_k;
			m_k = 945.0 / (8 * M_PI*h9);
			Real rl = r.norm();
			if (rl >= 0 && rl <= m_radius)
			{
				Real h2 = m_radius*m_radius;
				Real r2 = rl*rl;
				Real hr = h2 - r2;
				res = m_k*hr*hr*(r2 - 0.75*hr);
			}
			else
			{
				res = 0;
			}
			return res;
		}

	};
}

#endif
