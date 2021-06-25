#include "ParticleFluid.h"
#include "PositionBasedFluidModel.h"

#include "Framework/Topology/PointSet.h"
#include "Core/Utility.h"
#include "SummationDensity.h"
#include <fstream>
#include <time.h>

using namespace std;

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
		std::vector<std::shared_ptr<ParticleEmitter<TDataType>>> m_particleEmitters = this->getParticleEmitters();

		int total_num = 0;
		
		//std::cout << "HHH " << m_particleEmitters.size() <<" "<< this->currentPosition()->getElementCount()<< std::endl;
		//DeviceArray<Coord>& position = this->currentPosition()->getValue();
		//HostArray<Coord> h_position(this->currentPosition()->getElementCount());
		//Function1Pt::copy(h_position, position);
		//std::cout << "h_position " << h_position[0][0] << " " << h_position[0][1] << " " << h_position[0][2] << std::endl;
		//{//write to obj
		//	static int frame_id = 0;
		//	const std::string filename =  "particle_frame_" + std::to_string(frame_id) + ".dat";
		//	std::ofstream ofs(filename);
		//	for (size_t i = 0; i < h_position.size(); ++i) {
		//		ofs << h_position[i][0] << " " << h_position[i][1] << " " << h_position[i][2] << std::endl;
		//	}
		//	frame_id++;
		//}
	
		if (m_particleEmitters.size() > 0)
		{
			int total_num = this->currentPosition()->getElementCount();

			if (total_num > 0)
			{
				DeviceArray<Coord>& position = this->currentPosition()->getValue();
				DeviceArray<Coord>& velocity = this->currentVelocity()->getValue();
				DeviceArray<Coord>& force = this->currentForce()->getValue();

				int start = 0;
				for (int i = 0; i < m_particleEmitters.size(); i++)
				{
					int num = m_particleEmitters[i]->currentPosition()->getElementCount();
					if (num > 0)
					{
						auto points = m_particleEmitters[i]->currentPosition()->getValue();
						auto vels = m_particleEmitters[i]->currentVelocity()->getValue();
						auto fors = m_particleEmitters[i]->currentForce()->getValue();

						cudaMemcpy(points.getDataPtr(), position.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						cudaMemcpy(vels.getDataPtr(), velocity.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						cudaMemcpy(fors.getDataPtr(), force.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						start += num;
						// 						if (rand() % 1 == 0)
						// 							m_particleEmitters[i]->advance2(this->getDt());
					}
				}
			}
		}


		for (int i = 0; i < m_particleEmitters.size(); i++)
		{
			m_particleEmitters[i]->advance2(this->getDt());
		}

		total_num = 0;
		if (m_particleEmitters.size() > 0)
		{
			for (int i = 0; i < m_particleEmitters.size(); i++)
			{
				total_num += m_particleEmitters[i]->currentPosition()->getElementCount();
			}

			if (total_num > 0)
			{
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
					int num = m_particleEmitters[i]->currentPosition()->getElementCount();
					if (num > 0)
					{
						DeviceArray<Coord>& points = m_particleEmitters[i]->currentPosition()->getValue();
						DeviceArray<Coord>& vels = m_particleEmitters[i]->currentVelocity()->getValue();
						DeviceArray<Coord>& fors = m_particleEmitters[i]->currentForce()->getValue();

						cudaMemcpy(position.getDataPtr() + start, points.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						cudaMemcpy(velocity.getDataPtr() + start, vels.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						cudaMemcpy(force.getDataPtr() + start, fors.getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						start += num;
					}
				}
			}
		}
		else
		{
			total_num = this->currentPosition()->getElementCount();
		}

		std::cout << "Total number: " << total_num << std::endl;

		if (total_num > 0 && this->self_update)
		{
			auto nModel = this->getNumericalModel();
			nModel->step(this->getDt());
		}

		//printf("%d\n", this->currentPosition()->getElementCount());

		
	}


	template<typename TDataType>
	bool PhysIKA::ParticleFluid<TDataType>::resetStatus()
	{
		//printf("reset fluid\n");
		std::vector<std::shared_ptr<ParticleEmitter<TDataType>>> m_particleEmitters = this->getParticleEmitters();
		if(m_particleEmitters.size() > 0)
		{ 
			this->currentPosition()->setElementCount(0);
			this->currentVelocity()->setElementCount(0);
			this->currentForce()->setElementCount(0);
		}
		else 
			return ParticleSystem<TDataType>::resetStatus();
		return true;
	}

}