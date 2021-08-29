/**
 * @author     : Xiaowei He (xiaowei@iscas.ac.cn)
 * @date       : 2020-10-07
 * @description: Declaration of DensityPBD class, which implements the position-based fluids
 *               For more details, please refer to [Micklin et al. 2013] "Position Based Fluids"
 * @version    : 1.0
 */
#pragma once
#include "Core/Array/Array.h"
#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Kernel.h"

namespace PhysIKA {

template <typename TDataType>
class SummationDensity;

/*!
    *    \class    DensityPBD
    *    \brief    This class implements a position-based solver for incompressibility.
    */
template <typename TDataType>
class DensityPBD : public ConstraintModule
{
    DECLARE_CLASS_1(DensityPBD, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    DensityPBD();
    ~DensityPBD() override;

    /**
     * enforce density constraint
     * inPosition and inVelocity should be initialized before calling this API
     */
    bool constrain() override;
    /**
     * take one iteration using PBD
     */
    void takeOneIteration();
    /**
     * update velocity after PBD solving
     */
    void updateVelocity();

public:
    DeviceArrayField<Real> m_massInv;  // mass^-1 as described in unified particle physics

public:
    DEF_EMPTY_VAR(IterationNumber, int, "Iteration number of the PBD solver");

    DEF_EMPTY_VAR(RestDensity, Real, "Reference density");
    /**
         * @brief initial sampling distance of fluid particles
         */
    DEF_EMPTY_VAR(SamplingDistance, Real, "");
    /**
            * @brief smoothing length
            * A positive number represents the radius of neighborhood for each point
            */
    DEF_EMPTY_VAR(SmoothingLength, Real, "");

    /**
         * @brief Particle positions
         */
    DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "Input particle position");

    /**
         * @brief Particle velocities
         */
    DEF_EMPTY_IN_ARRAY(Velocity, Coord, DeviceType::GPU, "Input particle velocity");

    /**
         * @brief Neighboring particles' ids
         *
         */
    DEF_EMPTY_IN_NEIGHBOR_LIST(NeighborIndex, int, "Neighboring particles' ids");

    /**
         * @brief New particle positions
         */
    DEF_EMPTY_OUT_ARRAY(Position, Coord, DeviceType::GPU, "Output particle position");

    /**
         * @brief New particle velocities
         */
    DEF_EMPTY_OUT_ARRAY(Velocity, Coord, DeviceType::GPU, "Output particle velocity");

    /**
         * @brief Final particle densities
         */
    DEF_EMPTY_OUT_ARRAY(Density, Real, DeviceType::GPU, "Final particle density");

private:
    SpikyKernel<Real> m_kernel;

    DeviceArray<Real>  m_lamda;     //!< the lambda in eq 11
    DeviceArray<Coord> m_deltaPos;  //!< the delta p in eq 14
    DeviceArray<Coord> m_position_old;

private:
    std::shared_ptr<SummationDensity<TDataType>> m_summation;
};

}  // namespace PhysIKA