#include "ParticleFluid.h"
#include "PositionBasedFluidModel.h"

#include "Framework/Topology/PointSet.h"
#include "Rendering/PointRenderModule.h"
#include "Core/Utility.h"


namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleFluid, TDataType)

	template<typename TDataType>
	ParticleFluid<TDataType>::ParticleFluid(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		auto pbf = std::make_shared<PositionBasedFluidModel<TDataType>>();
		this->setNumericalModel(pbf);

		this->getPosition()->connect(pbf->m_position);
		this->getVelocity()->connect(pbf->m_velocity);
		this->getForce()->connect(pbf->m_forceDensity);

//		this->getVelocity()->connect(this->getRenderModule()->m_vecIndex);
		
				//auto fluid = std::make_shared<PositionBasedFluidModel<TDataType>>();
//		auto fluid = std::make_shared<MultifluidModel<TDataType>>();
//		this->setNumericalModel(fluid);
//		this->getPosition()->connect(fluid->m_position);
//		this->getVelocity()->connect(fluid->m_velocity);
//		this->getForce()->connect(fluid->m_forceDensity);
//		this->getColor()->connect(fluid->m_color);
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