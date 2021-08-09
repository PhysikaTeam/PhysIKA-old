/**
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-9
 * @description: Declaration of SemiAnalyticalSFINode class, which is a container for fluids with semi-analytical boundaries
 * @version    : 1.1
 */
#pragma once
#include "Framework/Framework/Node.h"
#include "Framework/Framework/DeclareNodeField.h"
#include "Framework/Framework/ModuleTopology.h"

#include "SemiAnalyticalIncompressibleFluidModel.h"
#include "PositionBasedFluidModelMesh.h"
#include "TriangularSurfaceMeshNode.h"

namespace PhysIKA {
class Attribute;
template <typename T>
class RigidBody;
template <typename T>
class ParticleSystem;
template <typename T>
class TriangularSurfaceMeshNode;
template <typename T>
class NeighborQuery;
template <typename TDataType>
class PointSet;
template <typename TDataType>
class TriangleSet;
/**
 * SemiAnalyticalSFINode
 * a scene node for fluids with semi-analytical boundaries
 * The default solver is PBD
 * reference: "Position Based Fluids", "A Variational Staggered Particle Framework for Incompressible Free-Surface Flows" and
 * "Semi-analytical Solid Boundary Conditions for Free Surface Flows"
 *
 * The source of fluids and boundaries can be setup exclusively by calling 
 * addParticleSystem and addTriangularSurfaceMeshNode()
 *
 */

template <typename TDataType>
class SemiAnalyticalSFINode : public Node
{
    DECLARE_CLASS_1(SemiAnalyticalSFINode, TDataType)
public:
    typedef typename TDataType::Real          Real;
    typedef typename TDataType::Coord         Coord;
    typedef typename TopologyModule::Triangle Triangle;

    SemiAnalyticalSFINode(std::string name = "SemiAnalyticalSFINode");
    ~SemiAnalyticalSFINode() override;

public:
    bool initialize() override;

    

    bool resetStatus() override;

    void advance(Real dt) override;

    void setInteractionDistance(Real d);

    //returns the particle position
    DeviceArrayField<Coord>* getParticlePosition()
    {
        return &m_particle_position;
    }
    //returns the particle arrtribute
    DeviceArrayField<Attribute>* getParticleAttribute()
    {
        return &m_particle_attribute;
    }
    //returns the particle velocity
    DeviceArrayField<Coord>* getParticleVelocity()
    {
        return &m_particle_velocity;
    }
    //returns the force density
    DeviceArrayField<Coord>* getParticleForceDensity()
    {
        return &m_particle_force_density;
    }
    //returns the particle mass
    DeviceArrayField<Real>* getParticleMass()
    {
        return &m_particle_mass;
    }

    //         DeviceArrayField<int>* getParticleId()
    //         {
    //             return &ParticleId;
    //         }
    //returns the trianlg vertex
    DeviceArrayField<Coord>* getTriangleVertex()
    {
        return &m_triangle_vertex;
    }
    //returns the triangle vertex in last time step, reserved for CCDs
    DeviceArrayField<Coord>* getTPO()
    {
        return &m_triangle_vertex_old;
    }
    //returns the triangle index
    DeviceArrayField<Triangle>* getTriangleIndex()
    {
        return &m_triangle_index;
    }
    //returns the triangle mass
    DeviceArrayField<Real>* getTriangleVertexMass()
    {
        return &m_triangle_vertex_mass;
    }

private:
    DEF_NODE_PORTS(ParticleSystem, ParticleSystem<TDataType>, "Particle Systems");
    DEF_NODE_PORTS(TriangularSurfaceMeshNode, TriangularSurfaceMeshNode<TDataType>, "Triangular Surface Mesh Node");

    VarField<Real> radius;

    DeviceArrayField<Real>      m_particle_mass;
    DeviceArrayField<Coord>     m_particle_velocity;
    DeviceArrayField<Coord>     m_particle_position;
    DeviceArrayField<Attribute> m_particle_attribute;

    DeviceArrayField<Real>     m_triangle_vertex_mass;
    DeviceArrayField<Coord>    m_triangle_vertex;
    DeviceArrayField<Coord>    m_triangle_vertex_old;
    DeviceArrayField<Triangle> m_triangle_index;

    DeviceArrayField<Coord> m_particle_force_density;

    //        DeviceArrayField<int> m_fixed;
    //        DeviceArrayField<Coord> BoundryForce;
    //        DeviceArrayField<Coord> ElasityForce;
    //        DeviceArrayField<Real> ElasityPressure;

    DeviceArray<int>      m_objId;
    DeviceArrayField<int> ParticleId;

    DeviceArray<Coord> posBuf;
    DeviceArray<Coord> VelBuf;

    DeviceArray<Real>  weights;
    DeviceArray<Coord> init_pos;

    //std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;
    //std::vector<std::shared_ptr<TriangularSurfaceMeshNode<TDataType>>> m_surfaces;
};

#ifdef PRECISION_FLOAT
template class SemiAnalyticalSFINode<DataType3f>;
#else
template class SolidFluidInteractionTmp<DataType3d>;
#endif
}  // namespace PhysIKA