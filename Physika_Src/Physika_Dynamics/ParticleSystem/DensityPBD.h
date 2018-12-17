#pragma once
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Framework/Framework/ModuleConstraint.h"

namespace Physika {

	template<typename TDataType> class DensitySummation;

	/*!
	*	\class	DensityPBD
	*	\brief	This class implements a position-based solver for incompressibility.
	*/
	template<typename TDataType>
	class DensityPBD : public ConstraintModule
	{
		DECLARE_CLASS_1(DensityPBD, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DensityPBD();
		~DensityPBD() override;

		bool constrain() override;

		void setMassID(FieldID id) { m_massID = id; }
		void setPositionID(FieldID id) { m_posID = id; }
		void setVelocityID(FieldID id) { m_velID = id; }
		void setNeighborhoodID(FieldID id) {m_neighborhoodID = id; }

		void setIterationNumber(int n) { m_maxIteration = n; }
		void setSmoothingLength(Real len) { m_smoothingLength = len; }

	protected:
		bool initializeImpl() override;

	protected:
		FieldID m_massID;
		FieldID m_posID;
		FieldID m_velID;
		FieldID m_neighborhoodID;

	private:
		int m_maxIteration;
		Real m_smoothingLength;

		DeviceArray<Real> m_rhoArr;
		DeviceArray<Real> m_lamda;
		DeviceArray<Coord> m_deltaPos;

		std::shared_ptr<DensitySummation<TDataType>> m_densitySum;
	};


#ifdef PRECISION_FLOAT
	template class DensityPBD<DataType3f>;
#else
 	template class DensityPBD<DataType3d>;
#endif
}