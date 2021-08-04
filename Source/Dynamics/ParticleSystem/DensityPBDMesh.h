/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-08-04
 * @description: Declaration of DensityPBDMesh class, which implements the position-based part of semi-analytical boundary conditions
 *               introduced in the paper <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * @version    : 1.1
 */

#pragma once
#include "Core/Array/Array.h"
#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Kernel.h"
#include "Framework/Framework/ModuleTopology.h"

namespace PhysIKA {
/**
 * DensityPBDMesh implements the position-based part of semi-analytical boundary conditions of the paper
 * <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * It is used in PositionBasedFluidModelMesh class
 */

template <typename TDataType>
class DensitySummationMesh;


template <typename TDataType>
class DensityPBDMesh : public ConstraintModule
{
    DECLARE_CLASS_1(DensityPBDMesh, TDataType)
public:
    typedef typename TDataType::Real          Real;
    typedef typename TDataType::Coord         Coord;
    typedef typename TopologyModule::Triangle Triangle;

    DensityPBDMesh();
    ~DensityPBDMesh() override;

    bool constrain() override;

    void takeOneIteration();

    void updateVelocity();

    void setIterationNumber(int n)
    {
        m_maxIteration = n;
    }

    DeviceArray<Real>& getDensity()
    {
        return m_density.getValue();
    }

protected:
    bool initializeImpl() override;

public:
    VarField<Real> m_restDensity;

     /**
            * @brief smoothing length
            * A positive number represents the radius of neighborhood for each point
            */
    VarField<Real> m_smoothingLength;

     /**
         * @brief Particle position
         */
    DeviceArrayField<Coord> m_position;
    /**
         * @brief Particle velocity
         */
    DeviceArrayField<Coord> m_velocity;
    DeviceArrayField<Real>  m_massInv;  
     /**
         * @brief Particle velocity norm
         */
    DeviceArrayField<Real>  m_veln;

     /**
         * @brief neighbor list of particles, only neighbor pairs of particle-particle are counted
         */
    NeighborField<int> m_neighborhood;
    /**
         * @brief neighbor list of particles and mesh triangles, only neighbor pairs of particle-triangle are counted
         */
    NeighborField<int>         m_neighborhoodTri;
    /**
         * @brief positions of Triangle vertexes
         */
    DeviceArrayField<Coord>    TriPoint;
    /**
         * @brief Triangle indexes, represented by three integers, indicating the three indexes of triangle vertex
         */
    DeviceArrayField<Triangle> Tri;
    /**
         * @brief array of density, the output of DensitySummationMesh
         */
    DeviceArrayField<Real> m_density;
    /**
         * @brief initial sampling distance of fluid particles
         */
    VarField<Real> sampling_distance;

    VarField<int>  use_mesh;
    VarField<int>  use_ghost;

    VarField<int> Start;

private:
    int m_maxIteration;

    SpikyKernel<Real> m_kernel;

    DeviceArray<Real>  m_lamda;
    DeviceArray<Coord> m_deltaPos;
    DeviceArray<Coord> m_position_old;

    std::shared_ptr<DensitySummationMesh<TDataType>> m_densitySum;
};

}  // namespace PhysIKA