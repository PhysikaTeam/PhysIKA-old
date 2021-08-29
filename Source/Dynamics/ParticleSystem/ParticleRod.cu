/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-12-22
 * @description: Implementation of ParticleRod class, projective-peridynamics based elastic rod
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-23
 * @description: poslish code, fix bugs
 * @version    : 1.1
 */

#include "ParticleRod.h"

#include "Core/Utility.h"
#include "Framework/Topology/PointSet.h"
#include "ParticleIntegrator.h"
#include "OneDimElasticityModule.h"
#include "FixedPoints.h"
#include "SimpleDamping.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(ParticleRod, TDataType)

template <typename TDataType>
ParticleRod<TDataType>::ParticleRod(std::string name)
    : ParticleSystem<TDataType>(name)
{
    m_horizon.setValue(0.008);
    this->attachField(&m_horizon, "horizon", "horizon");

    m_stiffness.setValue(0.5);
    this->attachField(&m_stiffness, "stiffness", "stiffness");

    m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
    this->currentPosition()->connect(m_integrator->inPosition());
    this->currentVelocity()->connect(m_integrator->inVelocity());
    this->currentForce()->connect(m_integrator->inForceDensity());

    m_one_dim_elasticity = this->template addConstraintModule<OneDimElasticityModule<TDataType>>("elasticity module");
    this->currentPosition()->connect(&m_one_dim_elasticity->m_position);
    this->currentVelocity()->connect(&m_one_dim_elasticity->m_velocity);
    m_horizon.connect(&m_one_dim_elasticity->m_distance);
    m_mass.connect(&m_one_dim_elasticity->m_mass);
    m_one_dim_elasticity->setIterationNumber(10);

    m_fixed = this->template addConstraintModule<FixedPoints<TDataType>>("fixed");
    this->currentPosition()->connect(&m_fixed->m_position);
    this->currentVelocity()->connect(&m_fixed->m_velocity);

    m_damping = this->template addConstraintModule<SimpleDamping<TDataType>>("damping");
    this->currentVelocity()->connect(&m_damping->m_velocity);
}

template <typename TDataType>
ParticleRod<TDataType>::~ParticleRod()
{
}

template <typename TDataType>
void ParticleRod<TDataType>::setParticles(std::vector<Coord> particles)
{
    m_pSet->setPoints(particles);
}

template <typename TDataType>
void ParticleRod<TDataType>::setMaterialStiffness(Real stiffness)
{
    m_one_dim_elasticity->setMaterialStiffness(stiffness);
}

template <typename TDataType>
bool ParticleRod<TDataType>::initialize()
{
    ParticleSystem<TDataType>::initialize();

    auto& list = this->getModuleList();
    for (auto iter = list.begin(); iter != list.end(); iter++)
    {
        (*iter)->initialize();
    }

    return true;
}

template <typename TDataType>
bool ParticleRod<TDataType>::resetStatus()
{
    ParticleSystem<TDataType>::resetStatus();
    resetMassField();
    m_modified = false;

    return true;
}

template <typename TDataType>
void ParticleRod<TDataType>::resetMassField()
{
    int num = this->currentPosition()->getElementCount();
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

template <typename TDataType>
void ParticleRod<TDataType>::addFixedParticle(int id, Coord pos)
{
    m_fixed->addFixedPoint(id, pos);
    m_fixedIds.push_back(id);
    m_modified = true;
}

template <typename TDataType>
void ParticleRod<TDataType>::removeFixedParticle(int id)
{
    m_fixed->removeFixedPoint(id);

    for (auto it = m_fixedIds.begin(); it != m_fixedIds.end();)
    {
        if (*it == id)
        {
            m_fixedIds.erase(it);
        }
        else
        {
            it++;
        }
    }

    m_modified = true;
}

template <typename TDataType>
void ParticleRod<TDataType>::doCollision(Coord pos, Coord dir)
{
    m_fixed->constrainPositionToPlane(pos, dir);
}

template <typename TDataType>
void ParticleRod<TDataType>::removeAllFixedPositions()
{
    m_fixed->clear();
    m_fixedIds.clear();
    m_modified = true;
}

template <typename TDataType>
void ParticleRod<TDataType>::getHostPosition(std::vector<Coord>& pos)
{
    int pNum = this->currentPosition()->getValue().size();
    if (pos.size() != pNum)
    {
        pos.resize(pNum);
    }

    cudaMemcpy(&pos[0], this->currentPosition()->getValue().getDataPtr(), pNum * sizeof(Coord), cudaMemcpyDeviceToHost);
}

template <typename TDataType>
void ParticleRod<TDataType>::setDamping(Real d)
{
    m_damping->setDampingCofficient(d);
}

template <typename TDataType>
void ParticleRod<TDataType>::advance(Real dt)
{
    if (m_modified == true)
    {
        resetMassField();
        m_modified = false;
    }

    if (m_fixed != nullptr)
        m_fixed->constrain();

    if (m_integrator != nullptr)
    {
        m_integrator->begin();
        m_integrator->integrate();
    }

    if (m_one_dim_elasticity != nullptr)
        m_one_dim_elasticity->constrain();

    if (m_damping != nullptr)
        m_damping->constrain();

    if (m_integrator != nullptr)
        m_integrator->end();
}

}  // namespace PhysIKA