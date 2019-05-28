#include "StaticBoundary.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Framework/Framework/Log.h"
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Core/Utilities/CudaRand.h"
#include "Physika_Dynamics/RigidBody/RigidBody.h"
#include "Physika_Dynamics/ParticleSystem/ParticleSystem.h"
#include "Physika_Dynamics/ParticleSystem/BoundaryConstraint.h"

#include "Physika_Geometry/SDF/DistanceField3D.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(StaticBoundary, TDataType)

	template<typename TDataType>
	StaticBoundary<TDataType>::StaticBoundary()
		: Node()
	{
	}

	template<typename TDataType>
	StaticBoundary<TDataType>::~StaticBoundary()
	{
	}

	template<typename TDataType>
	bool StaticBoundary<TDataType>::addRigidBody(std::shared_ptr<RigidBody<TDataType>> child)
	{
		this->addChild(child);
		m_rigids.push_back(child);
		return true;
	}

	template<typename TDataType>
	bool StaticBoundary<TDataType>::addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child)
	{
		this->addChild(child);
		m_particleSystems.push_back(child);
		return true;
	}

	template<typename TDataType>
	void StaticBoundary<TDataType>::advance(Real dt)
	{
		for (size_t t = 0; t < m_obstacles.size(); t++)
		{

			for (int i = 0; i < m_particleSystems.size(); i++)
			{
				DeviceArrayField<Coord>* posFd = m_particleSystems[i]->getPosition();
				DeviceArrayField<Coord>* velFd = m_particleSystems[i]->getVelocity();
				m_obstacles[t]->constrain(posFd->getValue(), velFd->getValue(), dt);
			}
		}
	}


	template<typename TDataType>
	void StaticBoundary<TDataType>::loadSDF(std::string filename, bool bOutBoundary)
	{
		auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
		boundary->load(filename);

		if (bOutBoundary)
		{
			boundary->invertSDF();
		}

		m_obstacles.push_back(boundary);
	}


	template<typename TDataType>
	void StaticBoundary<TDataType>::loadCube(Coord lo, Coord hi, bool bOutBoundary /*= false*/)
	{
		auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
		boundary->setCube(lo, hi);

		if (bOutBoundary)
		{
			boundary->invertSDF();
		}

		m_obstacles.push_back(boundary);
	}


	template<typename TDataType>
	void StaticBoundary<TDataType>::loadShpere(Coord center, Real r, bool bOutBoundary /*= false*/)
	{
		auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
		boundary->setSphere(center, r);

		if (bOutBoundary)
		{
			boundary->invertSDF();
		}

		m_obstacles.push_back(boundary);
	}


	template<typename TDataType>
	void StaticBoundary<TDataType>::scale(Real s)
	{
		for (int i = 0; i < m_obstacles.size(); i++)
		{
			m_obstacles[i]->m_cSDF->Scale(s);
		}
	}

	template<typename TDataType>
	void StaticBoundary<TDataType>::translate(Coord t)
	{
		for (int i = 0; i < m_obstacles.size(); i++)
		{
			m_obstacles[i]->m_cSDF->Translate(t);
		}
	}
}