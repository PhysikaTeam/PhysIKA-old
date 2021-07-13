#pragma once
#include "Dynamics/ParticleSystem/ParticleSystem.h"
#include "Dynamics/ParticleSystem/ParticleEmitter.h"

namespace PhysIKA
{
    /*!
    *    \class    ParticleFluid
    *    \brief    Position-based fluids.
    *
    *    This class implements a position-based fluid solver.
    *    Refer to Macklin and Muller's "Position Based Fluids" for details
    *
    */
    template<typename TDataType>
    class ParticleFluidFast : public ParticleSystem<TDataType>
    {
        DECLARE_CLASS_1(ParticleFluid, TDataType)
    public:
        typedef typename TDataType::Real Real;
        typedef typename TDataType::Coord Coord;

        ParticleFluidFast(std::string name = "default");
        virtual ~ParticleFluidFast();

        void advance(Real dt) override;
        bool resetStatus() override;

        /**
         * @brief Particle position
         */
        DEF_EMPTY_CURRENT_ARRAY(PositionInOrder, Coord, DeviceType::GPU, "Particle position");


        /**
         * @brief Particle velocity
         */
        DEF_EMPTY_CURRENT_ARRAY(VelocityInOrder, Coord, DeviceType::GPU, "Particle velocity");

        /**
         * @brief Particle force
         */
        DEF_EMPTY_CURRENT_ARRAY(ForceInOrder, Coord, DeviceType::GPU, "Force on each particle");

        DeviceArray<int> ids;
        DeviceArray<int> idsInOrder;

    private:
        DEF_NODE_PORTS(ParticleEmitter, ParticleEmitter<TDataType>, "Particle Emitters");
    };

#ifdef PRECISION_FLOAT
    template class ParticleFluidFast<DataType3f>;
#else
    template class ParticleFluidFast<DataType3d>;
#endif
}