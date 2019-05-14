#include "ParticleElasticBody.h"
#include "PositionBasedFluidModel.h"

#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Render/PointRenderModule.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Peridynamics.h"


namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleElasticBody, TDataType)

	template<typename TDataType>
	ParticleElasticBody<TDataType>::ParticleElasticBody(std::string name)
		: ParticleSystem(name)
	{
		auto peri = std::make_shared<Peridynamics<TDataType>>();
		this->setNumericalModel(peri);
		getPosition()->connect(peri->m_position);
		getVelocity()->connect(peri->m_velocity);
		getForce()->connect(peri->m_forceDensity);
	}

	template<typename TDataType>
	ParticleElasticBody<TDataType>::~ParticleElasticBody()
	{
		
	}

	template<typename TDataType>
	void ParticleElasticBody<TDataType>::advance(Real dt)
	{
		auto nModel = this->getNumericalModel();
		nModel->step(this->getDt());
	}
}