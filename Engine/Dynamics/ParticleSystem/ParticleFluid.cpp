#include "ParticleFluid.h"
#include "PositionBasedFluidModel.h"

#include "Framework/Topology/PointSet.h"
#include "Rendering/PointRenderModule.h"
#include "Core/Utility.h"
#include "DensitySummation.h"


namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleFluid, TDataType)

	template<typename TDataType>
	ParticleFluid<TDataType>::ParticleFluid(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		auto pbf = this->template setNumericalModel<PositionBasedFluidModel<TDataType>>("pbd");
		this->setNumericalModel(pbf);

		this->getPosition()->connect(pbf->m_position);
		this->getVelocity()->connect(pbf->m_velocity);
		this->getForce()->connect(pbf->m_forceDensity);
	}

	template<typename TDataType>
	ParticleFluid<TDataType>::~ParticleFluid()
	{
		
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::advance(Real dt)
	{
// 		auto pbf = this->getModule<PositionBasedFluidModel<TDataType>>("pbd");
// 
// 		pbf->getDensityField()->connect(this->getRenderModule()->m_scalarIndex);
// 		this->getRenderModule()->setColorRange(950, 1100);
// 		this->getRenderModule()->setReferenceColor(1000);

		auto nModel = this->getNumericalModel();
		nModel->step(this->getDt());
	}
}