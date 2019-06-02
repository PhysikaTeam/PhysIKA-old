#pragma once


#include "Physika_Core/Cuda_Array/Array.h"

#include "Physika_Framework/Framework/Base.h"
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Framework/Framework/NumericalModel.h"
#include "Physika_Framework/Framework/Module.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"

namespace Physika
{
    template<typename TDataType, int PhaseCount = 2>
	class CahnHilliard : public Module
    {
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
        using PhaseVector = Vector<Real, PhaseCount>;

		CahnHilliard();
		~CahnHilliard() override;

		bool initializeImpl() override;

		bool integrate();

		std::string getModuleType() override { return "NumericalIntegrator"; }

        NeighborField<int> m_neighborhood;

		VarField<Real> m_particleVolume;
		VarField<Real> m_smoothingLength;

		VarField<Real> m_degenerateMobilityM;

        DeviceArrayField<Coord> m_position;
		DeviceArrayField<PhaseVector> m_chemicalPotential;
        DeviceArrayField<PhaseVector> m_concentration;
	};
#ifdef PRECISION_FLOAT
	template class CahnHilliard<DataType3f>;
#else
	template class CahnHilliard<DataType3d>;
#endif
}

