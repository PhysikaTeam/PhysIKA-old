#pragma once
#include "Physika_Framework/Framework/ModuleConstraint.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"

namespace Physika {
	template<typename TDataType>
	class ImplicitViscosity : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ImplicitViscosity();
		~ImplicitViscosity() override;
		
		bool constrain() override;

		void setNeighborhoodID(FieldID id) { m_neighborhoodID = id; }

		void setSmoothingLength(Real len) { m_smoothingLength = len; }
		void setIterationNumber(int n) { m_maxInteration = n; }

		void setViscosity(Real vis) { m_viscosity->setValue(vis); }

	protected:
		FieldID m_neighborhoodID;

	private:
		int m_maxInteration;
		Real m_smoothingLength;

		DeviceArray<Coord> m_velOld;
		DeviceArray<Coord> m_velBuf;

		std::shared_ptr<HostVarField<Real>> m_viscosity;
	};

#ifdef PRECISION_FLOAT
	template class ImplicitViscosity<DataType3f>;
#else
	template class ImplicitViscosity<DataType3d>;
#endif
}