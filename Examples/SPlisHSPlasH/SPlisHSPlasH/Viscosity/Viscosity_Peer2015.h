#ifndef __Viscosity_Peer2015_h__
#define __Viscosity_Peer2015_h__

#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/FluidModel.h"
#include "ViscosityBase.h"
#include "SPlisHSPlasH/Utilities/MatrixFreeSolver.h"


namespace SPH
{
	/** \brief This class implements the implicit simulation method for
	* viscous fluids introduced
	* by Peer et al. [PICT15].
	*
	* References:
	* - [PICT15] A. Peer, M. Ihmsen, J. Cornelis, and M. Teschner. An Implicit Viscosity Formulation for SPH Fluids. ACM Trans. Graph., 34(4):1-10, 2015. URL: http://doi.acm.org/10.1145/2766925
	*/
	class Viscosity_Peer2015 : public ViscosityBase
	{
	protected: 
		std::vector<Real> m_density;
		std::vector<Matrix3r> m_targetNablaV;
		typedef Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper, JacobiPreconditioner1D> Solver;
		Solver m_solver;
		unsigned int m_iterations;
		unsigned int m_maxIter;
		Real m_maxError;

		virtual void initParameters();
		void computeDensities();

	public:
		static int ITERATIONS;
		static int MAX_ITERATIONS;
		static int MAX_ERROR;

		Viscosity_Peer2015(FluidModel *model);
		virtual ~Viscosity_Peer2015(void);

		virtual void step();
		virtual void reset();

		virtual void performNeighborhoodSearchSort();

		static void matrixVecProd(const Real* vec, Real *result, void *userData);
		FORCE_INLINE static void diagonalMatrixElement(const unsigned int row, Real &result, void *userData);

		FORCE_INLINE const Matrix3r& getTargetNablaV(const unsigned int i) const
		{
			return m_targetNablaV[i];
		}

		FORCE_INLINE Matrix3r& getTargetNablaV(const unsigned int i)
		{
			return m_targetNablaV[i];
		}

		FORCE_INLINE void setTargetNablaV(const unsigned int i, const Matrix3r &val)
		{
			m_targetNablaV[i] = val;
		}
	};
}

#endif
