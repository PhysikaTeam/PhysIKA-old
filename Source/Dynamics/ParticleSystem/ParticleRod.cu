#include "ParticleRod.h"
#include "Framework/Topology/PointSet.h"
#include "Core/Utility.h"
#include "Framework/Topology/NeighborQuery.h"
#include "ParticleIntegrator.h"
#include "ElasticityModule.h"
#include "OneDimElasticityModule.h"
#include "FixedPoints.h"
#include "SimpleDamping.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(ParticleRod, TDataType)

	template<typename TDataType>
	ParticleRod<TDataType>::ParticleRod(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		m_horizon.setValue(0.008);
		this->attachField(&m_horizon, "horizon", "horizon");

		m_stiffness.setValue(0.5);
		this->attachField(&m_stiffness, "stiffness", "stiffness");

		m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		this->getPosition()->connect(m_integrator->m_position);
		this->getVelocity()->connect(m_integrator->m_velocity);
		this->getForce()->connect(m_integrator->m_forceDensity);

		auto m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
		m_horizon.connect(m_nbrQuery->m_radius);
		this->m_position.connect(m_nbrQuery->m_position);

// 		m_elasticity = this->template addConstraintModule<ElasticityModule<TDataType>>("elasticity");
// 		this->getPosition()->connect(m_elasticity->m_position);
// 		this->getVelocity()->connect(m_elasticity->m_velocity);
// 		m_horizon.connect(m_elasticity->m_horizon);
// 		m_nbrQuery->m_neighborhood.connect(m_elasticity->m_neighborhood);
// 		m_elasticity->setIterationNumber(10);

		m_one_dim_elasticity = this->template addConstraintModule<OneDimElasticityModule<TDataType>>("elasticity module");
		this->getPosition()->connect(m_one_dim_elasticity->m_position);
		this->getVelocity()->connect(m_one_dim_elasticity->m_velocity);
		m_horizon.connect(m_one_dim_elasticity->m_distance);
		m_mass.connect(m_one_dim_elasticity->m_mass);
		m_one_dim_elasticity->setIterationNumber(10);

		m_fixed = this->template addConstraintModule<FixedPoints<TDataType>>("fixed");
		this->getPosition()->connect(m_fixed->m_position);
		this->getVelocity()->connect(m_fixed->m_velocity);

		m_damping = this->template addConstraintModule<SimpleDamping<TDataType>>("damping");
		this->getVelocity()->connect(m_damping->m_velocity);
	}

	template<typename TDataType>
	ParticleRod<TDataType>::~ParticleRod()
	{
		
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::setParticles(std::vector<Coord> particles)
	{
		m_pSet->setPoints(particles);
	}


	template<typename TDataType>
	void ParticleRod<TDataType>::setLength(Real length)
	{
		m_horizon.setValue(length);
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::setMaterialStiffness(Real stiffness)
	{
// 		m_elasticity->setMu(0.01*stiffness);
// 		m_elasticity->setLambda(stiffness);
		m_one_dim_elasticity->setMaterialStiffness(stiffness);
	}


	template<typename TDataType>
	bool ParticleRod<TDataType>::initialize()
	{
		ParticleSystem<TDataType>::initialize();

		auto& list = this->getModuleList();
		std::list<std::shared_ptr<Module>>::iterator iter = list.begin();
		for (; iter != list.end(); iter++)
		{
			(*iter)->initialize();
		}

		return true;
	}

	template<typename TDataType>
	bool ParticleRod<TDataType>::resetStatus()
	{
		ParticleSystem<TDataType>::resetStatus();

		resetMassField();

		return true;
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::resetMassField()
	{
		int num = m_position.getElementCount();
		m_mass.setElementCount(num);

		std::vector<Real> host_mass;
		for (int i = 0; i < num; i++)
		{
			host_mass.push_back(Real(1));
		}

		for (int i = 0; i < m_fixedIds.size(); i++)
		{
			host_mass[m_fixedIds[i]] = Real(1000000);
		}

		m_mass.setValue(host_mass);
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::addFixedParticle(int id, Coord pos)
	{
		m_fixed->addFixedPoint(id, pos);

		m_fixedIds.push_back(id);

		m_modifed = true;
	}


	template<typename TDataType>
	void ParticleRod<TDataType>::removeFixedParticle(int id)
	{
		m_fixed->removeFixedPoint(id);
		
		for (auto it = m_fixedIds.begin(); it != m_fixedIds.end();) {
			if (*it == id) {
				m_fixedIds.erase(it);
			}
			else {
				it++;
			}
		}


		m_modifed = true;
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::doCollision(Coord pos, Coord dir)
	{
		m_fixed->constrainPositionToPlane(pos, dir);
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::removeAllFixedPositions()
	{
		m_fixed->clear();
		m_fixedIds.clear();
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::getHostPosition(std::vector<Coord>& pos)
	{
		int pNum = this->m_position.getValue().size();
		if (pos.size() != pNum)
		{
			pos.resize(pNum);
		}

		cudaMemcpy(&pos[0], m_position.getValue().getDataPtr(), pNum*sizeof(Coord), cudaMemcpyDeviceToHost);
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::setDamping(Real d)
	{
		m_damping->setDampingCofficient(d);
	}

	template<typename TDataType>
	void ParticleRod<TDataType>::advance(Real dt)
	{
		if (m_modifed == true)
		{
			resetMassField();
		}

		if (m_fixed != nullptr)
			m_fixed->constrain();

		if (m_integrator != nullptr)
			m_integrator->begin();

		m_integrator->integrate();
 	
		m_one_dim_elasticity->constrain();
		//m_elasticity->constrain();

		if (m_damping != nullptr)
			m_damping->constrain();

		if (m_integrator != nullptr)
			m_integrator->end();

	}

}