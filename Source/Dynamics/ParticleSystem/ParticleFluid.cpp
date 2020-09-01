#include "ParticleFluid.h"
#include "PositionBasedFluidModel.h"

#include "Framework/Topology/PointSet.h"
#include "Core/Utility.h"
#include "DensitySummation.h"


namespace PhysIKA
{
	IMPLEMENT_CLASS_1(ParticleFluid, TDataType)

	template<typename TDataType>
	ParticleFluid<TDataType>::ParticleFluid(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		auto pbf = this->template setNumericalModel<PositionBasedFluidModel<TDataType>>("pbd");
		this->setNumericalModel(pbf);

		this->currentPosition()->connect(&pbf->m_position);
		this->currentVelocity()->connect(&pbf->m_velocity);
		this->currentForce()->connect(&pbf->m_forceDensity);
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
		printf("%d\n", this->currentPosition()->getElementCount());

		//if (m_ParticleEmitter != NULL)
		//	m_ParticleEmitter->advance(this->getDt());
	}

	template<typename TDataType>
	bool ParticleFluid<TDataType>::addEmitter(std::shared_ptr<ParticleEmitter<TDataType>> child)
	{
		
		
		//m_ParticleEmitter = child;
		
	    //this->getParticleEmitters()->addNode(child.get());
		this->addParticleEmitter(child);
		//this->addChild(child);

		child->currentForce()->connect(this->currentForce());
		child->currentPosition()->connect(this->currentPosition());
		child->currentVelocity()->connect(this->currentVelocity());
		
		return true;
	}
}