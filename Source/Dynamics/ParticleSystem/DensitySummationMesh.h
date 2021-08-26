/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-08-04
 * @description: Declaration of DensitySummationMesh class, which implements density summation using semi-analytical boundary conditions
 *               introduced in the paper <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * @version    : 1.1
 */
#pragma once
#include "Framework/Framework/ModuleCompute.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Framework/Framework/ModuleTopology.h"

namespace PhysIKA {

/**
 * DensitySummationMesh calculates the density of a particle using semi-analytical boundary conditions of the paper
 * <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * It is used in DensityPBDMesh class
 */

template <typename TDataType>
class NeighborList;
template <typename TDataType>
class TriangleSet;

template <typename TDataType>
class DensitySummationMesh : public ComputeModule
{
    DECLARE_CLASS_1(DensitySummation, TDataType)

public:
    typedef typename TDataType::Real          Real;
    typedef typename TDataType::Coord         Coord;
    typedef typename TopologyModule::Triangle Triangle;

    DensitySummationMesh();
    ~DensitySummationMesh() override{};

    /**
     * handle the boundary conditions of fluids and mesh-based solid boundary
     * m_position&&m_neighborhood&&m_neighborhoodTri&&Tri&TriPoint need to be setup before calling this API
     * note that the other two comute() are not used as APIs in semi-analytical boundaries
     * @returns true
     */
    void compute() override;

    void compute(DeviceArray<Real>& rho);

    void compute(
        DeviceArray<Real>&                     rho,
        DeviceArray<Coord>&                    pos,
        DeviceArray<TopologyModule::Triangle>& Tri,
        DeviceArray<Coord>&                    positionTri,
        NeighborList<int>&                     neighbors,
        NeighborList<int>&                     neighborsTri,
        Real                                   smoothingLength,
        Real                                   mass,
        Real                                   sampling_distance,
        int                                    use_mesh,
        int                                    use_ghost,
        int                                    Start);

    void setCorrection(Real factor)
    {
        m_factor = factor;
    }
    void setSmoothingLength(Real length)
    {
        m_smoothingLength.setValue(length);
    }

protected:
    bool initializeImpl() override;

public:
    VarField<Real> m_mass;
    VarField<Real> m_restDensity;
    VarField<Real> m_smoothingLength;  //smoothing length, a positive number represents the radius of neighborhood for each point

    DeviceArrayField<Coord> m_position;  //particle positions
    DeviceArrayField<Real>  m_density;   //output, particle density

    NeighborField<int>         m_neighborhood;
    NeighborField<int>         m_neighborhoodTri;
    DeviceArrayField<Coord>    TriPoint;  //positions of the vertex of the triangle
    DeviceArrayField<Triangle> Tri;

    VarField<Real> sampling_distance;  //sampling distance of the particle
    VarField<int>  use_mesh;
    VarField<int>  use_ghost;
    VarField<int>  Start;

private:
    Real m_factor;  //!< a renormalization factor to transfer the weight calculated by the kernel function to density
};

#ifdef PRECISION_FLOAT
template class DensitySummationMesh<DataType3f>;
#else
template class DensitySummationMesh<DataType3d>;
#endif
}  // namespace PhysIKA