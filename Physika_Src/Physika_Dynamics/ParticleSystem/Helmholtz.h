#pragma once
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Framework/Framework/ModuleConstraint.h"

namespace Physika {

	template<typename TDataType> class DensitySummation;

	/*!
	*	\class	Helmholtz
	*	\brief	This class implements a position-based solver for incompressibility.
	*/
	template<typename TDataType>
	class Helmholtz : public ConstraintModule
	{
		DECLARE_CLASS_1(Helmholtz, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Helmholtz();
		~Helmholtz() override;

		bool constrain() override;

		void setPositionID(FieldID id) { m_posID = id; }
		void setVelocityID(FieldID id) { m_velID = id; }
		void setNeighborhoodID(FieldID id) {m_neighborhoodID = id; }

		void setIterationNumber(int n) { m_maxIteration = n; }
		void setSmoothingLength(Real len) { m_smoothingLength = len; }

		void computeC();
		void computeGC();
		void computeLC();

	protected:
		bool initializeImpl() override;

	protected:
		FieldID m_posID;
		FieldID m_velID;
		FieldID m_neighborhoodID;

	private:
		int m_maxIteration;
		Real m_smoothingLength;

		DeviceArray<Real> m_lamda;
		DeviceArray<Real> m_rho;
		DeviceArray<Coord> m_deltaPos;
		DeviceArray<Coord> m_originPos;
	};

#ifdef PRECISION_FLOAT
	template class Helmholtz<DataType3f>;
#else
 	template class Helmholtz<DataType3d>;
#endif
}