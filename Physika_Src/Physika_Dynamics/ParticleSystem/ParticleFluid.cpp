#include "ParticleFluid.h"
#include "PositionBasedFluidModel.h"

#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Render/PointRenderModule.h"
#include "Physika_Core/Utilities/Function1Pt.h"


namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleFluid, TDataType)

	template<typename TDataType>
	ParticleFluid<TDataType>::ParticleFluid(std::string name)
		: ParticleSystem(name)
	{
		auto pbf = std::make_shared<PositionBasedFluidModel<TDataType>>();
		this->setNumericalModel(pbf);
		getPosition()->connect(pbf->m_position);
		getVelocity()->connect(pbf->m_velocity);
		getForce()->connect(pbf->m_forceDensity);
	}

	template<typename TDataType>
	ParticleFluid<TDataType>::~ParticleFluid()
	{
		
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::advance(Real dt)
	{
		auto nModel = this->getNumericalModel();
		nModel->step(this->getDt());
	}
}