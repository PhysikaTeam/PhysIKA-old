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
		
		
		std::vector<std::shared_ptr<ParticleEmitter<TDataType>>> m_particleEmitters = this->getParticleEmitters();
		if(m_particleEmitters.size() > 0)
		{ 
			
			int total_num = 0;
			for (int i = 0; i < m_particleEmitters.size(); i++)
			{
				
				auto points = m_particleEmitters[i]->currentPosition()->getValue();
				total_num += points.size();
			//	printf("Emitter: %d Num: %d\n", i, points.size());

			}
			this->currentPosition()->setElementCount(total_num);
			this->currentVelocity()->setElementCount(total_num);
			this->currentForce()->setElementCount(total_num);

			//printf("###### %d\n", this->currentPosition()->getElementCount());

			DeviceArray<Coord>& position = this->currentPosition()->getValue();
			DeviceArray<Coord>& velocity = this->currentVelocity()->getValue();
			DeviceArray<Coord>& force = this->currentForce()->getValue();

			int start = 0;
			for (int i = 0; i < m_particleEmitters.size(); i++)
			{
				DeviceArray<Coord>& points = m_particleEmitters[i]->currentPosition()->getValue();
				DeviceArray<Coord>& vels = m_particleEmitters[i]->currentVelocity()->getValue();
				DeviceArray<Coord>& fors = m_particleEmitters[i]->currentForce()->getValue();
				int num = points.size();
				cudaMemcpy(position.getDataPtr() + start, points.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
				cudaMemcpy(velocity.getDataPtr() + start, vels.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
				cudaMemcpy(force.getDataPtr() + start, fors.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
				start += num;
			}
		}

		auto nModel = this->getNumericalModel();
		nModel->step(this->getDt());
		//printf("%d\n", this->currentPosition()->getElementCount());

		if (m_particleEmitters.size() > 0)
		{
			DeviceArray<Coord>& position = this->currentPosition()->getValue();
			DeviceArray<Coord>& velocity = this->currentVelocity()->getValue();
			DeviceArray<Coord>& force = this->currentForce()->getValue();

			int start = 0;
			for (int i = 0; i < m_particleEmitters.size(); i++)
			{
				DeviceArray<Coord>& points = m_particleEmitters[i]->currentPosition()->getValue();
				DeviceArray<Coord>& vels = m_particleEmitters[i]->currentVelocity()->getValue();
				DeviceArray<Coord>& fors = m_particleEmitters[i]->currentForce()->getValue();
				int num = points.size();
				cudaMemcpy(points.getDataPtr(), position.getDataPtr() + start ,num * sizeof(Coord), cudaMemcpyDeviceToDevice);
				cudaMemcpy(vels.getDataPtr(), velocity.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
				cudaMemcpy(fors.getDataPtr(), force.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
				start += num;
				
			}
		}

		//if (m_ParticleEmitter != NULL)
		//	m_ParticleEmitter->advance(this->getDt());
	}

	template<typename TDataType>
	bool ParticleFluid<TDataType>::addEmitter(std::shared_ptr<ParticleEmitter<TDataType>> child)
	{
		/*
		
		//m_ParticleEmitter = child;
		
	    //this->getParticleEmitters()->addNode(child.get());
		this->addParticleEmitter(child);
		//this->addChild(child);

		child->currentForce()->connect(this->currentForce());
		child->currentPosition()->connect(this->currentPosition());
		child->currentVelocity()->connect(this->currentVelocity());
		*/
		return true;
	}
}