#include "ParticleFluid.h"
//#include "PositionBasedFluidModel.h"
#include "MultifluidModel.h"

#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Render/PointRenderModule.h"
#include "Physika_Core/Utilities/Function1Pt.h"


namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleFluid, TDataType)

	template<typename TDataType>
	ParticleFluid<TDataType>::ParticleFluid(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		//auto fluid = std::make_shared<PositionBasedFluidModel<TDataType>>();
		auto fluid = std::make_shared<MultifluidModel<TDataType>>();
		this->setNumericalModel(fluid);
		this->getPosition()->connect(fluid->m_position);
		this->getVelocity()->connect(fluid->m_velocity);
		this->getForce()->connect(fluid->m_forceDensity);
		this->getColor()->connect(fluid->m_color);
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