#pragma once
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Framework/Framework/ModuleConstraint.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"

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

		void setIterationNumber(int n) { m_maxIteration = n; }

	protected:
		bool initializeImpl() override;

	public:
		VarField<Real> m_restDensity;
		VarField<Real> m_smoothingLength;

		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;

		DeviceArrayField<Real> m_density;

		DeviceArrayField<Real> m_massInv; // mass^-1 as described in unified particle physics

		NeighborField<int> m_neighborhood;

	private:
		int m_maxIteration;

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