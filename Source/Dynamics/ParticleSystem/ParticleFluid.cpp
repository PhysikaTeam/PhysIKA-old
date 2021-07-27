/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2018-12-17
 * @description: Implementation of ParticleFluid class, which is a container for particle-based fluid solvers
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-21
 * @description: poslish code
 * @version    : 1.1
 */

#include "ParticleFluid.h"

#include "Core/Utility.h"
#include "Framework/Topology/PointSet.h"
#include "PositionBasedFluidModel.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(ParticleFluid, TDataType)

template <typename TDataType>
ParticleFluid<TDataType>::ParticleFluid(std::string name)
    : ParticleSystem<TDataType>(name)
{
    auto pbf = this->template setNumericalModel<PositionBasedFluidModel<TDataType>>("pbd");
    this->setNumericalModel(pbf);

    this->currentPosition()->connect(&pbf->m_position);
    this->currentVelocity()->connect(&pbf->m_velocity);
    this->currentForce()->connect(&pbf->m_forceDensity);
}

template <typename TDataType>
ParticleFluid<TDataType>::~ParticleFluid()
{
}

template <typename TDataType>
void ParticleFluid<TDataType>::advance(Real dt)
{
    std::vector<std::shared_ptr<ParticleEmitter<TDataType>>> m_particleEmitters = this->getParticleEmitters();

    if (m_particleEmitters.size() > 0)
    {
        //update particle emitters' particle state with current simulation state
        int total_num = this->currentPosition()->getElementCount();

        if (total_num > 0)
        {
            DeviceArray<Coord>& position = this->currentPosition()->getValue();
            DeviceArray<Coord>& velocity = this->currentVelocity()->getValue();
            DeviceArray<Coord>& force    = this->currentForce()->getValue();

            int start = 0;
            for (int i = 0; i < m_particleEmitters.size(); i++)
            {
                int num = m_particleEmitters[i]->currentPosition()->getElementCount();
                if (num > 0)
                {
                    auto points = m_particleEmitters[i]->currentPosition()->getValue();
                    auto vels   = m_particleEmitters[i]->currentVelocity()->getValue();
                    auto fors   = m_particleEmitters[i]->currentForce()->getValue();

                    cudaMemcpy(points.getDataPtr(), position.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(vels.getDataPtr(), velocity.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(fors.getDataPtr(), force.getDataPtr() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
                    start += num;
                }
            }
        }
    }

    //apply particle emitters' state by applying emit rules
    //particle numbers may change after this call
    for (int i = 0; i < m_particleEmitters.size(); i++)
    {
        m_particleEmitters[i]->advance2(this->getDt());
    }

    int total_num = 0;
    if (m_particleEmitters.size() > 0)
    {
        for (int i = 0; i < m_particleEmitters.size(); i++)
        {
            total_num += m_particleEmitters[i]->currentPosition()->getElementCount();
        }

        //update simulation state with particle emitters' new state
        if (total_num > 0)
        {
            this->currentPosition()->setElementCount(total_num);
            this->currentVelocity()->setElementCount(total_num);
            this->currentForce()->setElementCount(total_num);

            DeviceArray<Coord>& position = this->currentPosition()->getValue();
            DeviceArray<Coord>& velocity = this->currentVelocity()->getValue();
            DeviceArray<Coord>& force    = this->currentForce()->getValue();

            int start = 0;
            for (int i = 0; i < m_particleEmitters.size(); i++)
            {
                int num = m_particleEmitters[i]->currentPosition()->getElementCount();
                if (num > 0)
                {
                    DeviceArray<Coord>& points = m_particleEmitters[i]->currentPosition()->getValue();
                    DeviceArray<Coord>& vels   = m_particleEmitters[i]->currentVelocity()->getValue();
                    DeviceArray<Coord>& fors   = m_particleEmitters[i]->currentForce()->getValue();

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

    std::cout << "Total number: " << total_num << std::endl;  //TODO(Zhu Fei): we should replace all std stream with LOG

    //apply simulation
    if (total_num > 0 && this->self_update)
    {
        auto nModel = this->getNumericalModel();
        nModel->step(this->getDt());
    }
}

template <typename TDataType>
bool PhysIKA::ParticleFluid<TDataType>::resetStatus()
{
    std::vector<std::shared_ptr<ParticleEmitter<TDataType>>> m_particleEmitters = this->getParticleEmitters();
    if (m_particleEmitters.size() > 0)
    {
        this->currentPosition()->setElementCount(0);
        this->currentVelocity()->setElementCount(0);
        this->currentForce()->setElementCount(0);
    }
    else
        return ParticleSystem<TDataType>::resetStatus();
    return true;
}

}  // namespace PhysIKA