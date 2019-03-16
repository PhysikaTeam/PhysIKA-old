#pragma once
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"
#include "Physika_Framework/Framework/ModuleForce.h"

namespace Physika {

	template<typename TDataType>
	class HyperelasticForce : public ForceModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TPair<TDataType> NPair;

		HyperelasticForce();
		~HyperelasticForce() override {};
		
		bool applyForce() override;

		void construct(NeighborList<int>& nbr, DeviceArray<Coord>& pos);
		bool isUpdateRequired() { return m_needUpdate; }

		void setHorizon(Real len) { m_horizon = len; }
	protected:
		FieldID m_posID;
		FieldID m_velID;
		FieldID m_forceID;
		FieldID m_initPosID;
		FieldID m_posPreID;
		FieldID m_neighborhoodID;

	private:
		bool m_needUpdate;
		Real m_horizon;

		NeighborList<NPair> m_refPos;

		DeviceArrayField<Real>* m_lambdas;
		DeviceArrayField<Real>* m_bulkCoef;
		DeviceArrayField<Coord>* m_tmpPos;
		DeviceArrayField<Coord>* m_accPos;
		DeviceArrayField<Matrix>* m_refMatrix;
	};


#ifdef PRECISION_FLOAT
	template class HyperelasticForce<DataType3f>;
#else
	template class HyperelasticForce<DataType3d>;
#endif
}