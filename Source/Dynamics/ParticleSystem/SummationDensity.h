/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Declaration of SummationDensity class, which calculates the densities of particles
 * @version    : 1.0
 *
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-08
 * @description: poslish code
 * @version    : 1.1
 */
#pragma once
#include "Framework/Framework/ModuleCompute.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"

namespace PhysIKA {

/**
     * @brief The standard summation density
     * 
     * @tparam TDataType 
     * 
     * Usage:
     * (1) initialize position and neighborlist by initializing inPosition and inNeighborList 
     * (2) call compute when needed, get the density from outDensity
     */
template <typename TDataType>
class SummationDensity : public ComputeModule
{
    DECLARE_CLASS_1(SummationDensity, TDataType)

public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    SummationDensity();
    ~SummationDensity() override{};

    void compute() override;

protected:
    //calculates a renormalization factor by using the given sampling distance
    void calculateScalingFactor();
    //calculates the mass of each particle using the given sampling distance
    void calculateParticleMass();

    /**
     * calculate the density
     *
     * @param[out]      rho          densities defined on particles 
     *
     * @calculates the densities and store them on rho
     */
    void compute(DeviceArray<Real>& rho);

    /**
     * calculate the density
     *
     * @param[out]      rho                densities defined on particles 
     * @param[in]       pos                particle positions
     * @param[in]       neighbors          neighbor list of particles 
     * @param[in]       smoothingLength    searching radius of particles 
     * @param[in]       mass               particle mass 
     * @calculates the densities and store them on rho
     */
    void compute(
        DeviceArray<Real>&  rho,
        DeviceArray<Coord>& pos,
        NeighborList<int>&  neighbors,
        Real                smoothingLength,
        Real                mass);

public:
    //the ideal density when all particles are uniformly distributed
    DEF_EMPTY_VAR(RestDensity, Real, "Rest Density");
    //the searching radius
    DEF_EMPTY_VAR(SmoothingLength, Real, "Indicating the smoothing length");
    //the sampling distance of particles, used to initialize the renormalization factor
    DEF_EMPTY_VAR(SamplingDistance, Real, "Indicating the initial sampling distance");

    ///Define inputs
    /**
         * @brief Particle positions
         */
    DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");

    /**
         * @brief Neighboring particles
         *
         */
    DEF_EMPTY_IN_NEIGHBOR_LIST(NeighborIndex, int, "Neighboring particles' ids");

    ///Define outputs
    /**
         * @brief Particle densities
         */
    DEF_EMPTY_OUT_ARRAY(Density, Real, DeviceType::GPU, "Particle position");

private:
    Real m_particle_mass;
    Real m_factor;
};

#ifdef PRECISION_FLOAT
template class SummationDensity<DataType3f>;
#else
template class SummationDensity<DataType3d>;
#endif
}  // namespace PhysIKA