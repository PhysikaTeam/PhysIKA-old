/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-08-06
 * @description: Declaration of SemiAnalyticalIncompressibleFluidModel class, a container for semi-analytical projection-based fluids 
 *               introduced in the paper <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * @version    : 1.1
 */
#pragma once
#include "Framework/Framework/NumericalModel.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "DensityPBD.h"
#include "SemiAnalyticalIncompressibilityModule.h"
#include "Attribute.h"
#include "Framework/Framework/ModuleTopology.h"
#include "MeshCollision.h"

/**
 * SemiAnalyticalIncompressibleFluidModel
 * a NumericalModel for semi-analytical projection-based fluids 
 * The solver is projection-based fluids with semi-analytical boundaries
 * reference: "Semi-analytical Solid Boundary Conditions for Free Surface Flows"
 *
 * Could be used by being created and initialized at SemiAnalyticalSFINode
 * Fields required to be initialized include:
 *      m_particle_position
 *      m_particle_velocity
 *      m_particle_force_density
 *      m_particle_attribute
 *      m_particle_mass
 *      m_triangle_vertex_mass
 *      m_triangle_index
 *      m_triangle_vertex
 *      m_triangle_vertex_old
 *      m_smoothing_length
 * 
 */

namespace PhysIKA {
template <typename TDataType>
class PointSetToPointSet;
template <typename TDataType>
class ParticleIntegrator;
template <typename TDataType>
class NeighborQuery;
template <typename TDataType>
class DensityPBD;
template <typename TDataType>
class MeshCollision;
template <typename TDataType>
class SurfaceTension;
template <typename TDataType>
class ImplicitViscosity;
template <typename TDataType>
class Helmholtz;
template <typename>
class PointSetToPointSet;
typedef typename TopologyModule::Triangle Triangle;

class ForceModule;
class ConstraintModule;
class Attribute;

template <typename TDataType>
class SemiAnalyticalIncompressibleFluidModel : public NumericalModel
{
    DECLARE_CLASS_1(SemiAnalyticalIncompressibleFluidModel, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    SemiAnalyticalIncompressibleFluidModel();
    virtual ~SemiAnalyticalIncompressibleFluidModel();

    void step(Real dt) override;

    /*
    * set smoothing lenth to be len
    */
    void setSmoothingLength(Real len)
    {
        m_smoothing_length.setValue(len);
    }
    /**
    *  currently have no influence on the behaviour
    */
    void setRestDensity(Real rho)
    {
        m_restRho = rho;
    }

    /**
    *  currently have no influence on the behaviour
    */
    void setIncompressibilitySolver(std::shared_ptr<ConstraintModule> solver);
    void setViscositySolver(std::shared_ptr<ConstraintModule> solver);
    void setSurfaceTensionSolver(std::shared_ptr<ConstraintModule> solver);

    /*
    *  have no infludence on behaviour, but can be used in visualizing densities
    */
    DeviceArrayField<Real>* getDensityField()
    {
        return m_pbdModule2->outDensity();
        //return m_force_density;
    }

public:
    VarField<Real> m_smoothing_length;  //searching distance for particles

    VarField<Real> max_vel;               //no use
    VarField<Real> var_smoothing_length;  //no use

    DeviceArrayField<Real> m_particle_mass;  //mass of particles

    DeviceArrayField<Coord> m_particle_position;  //particle positions
    DeviceArrayField<Coord> m_particle_velocity;  //particle velocities

    DeviceArrayField<Attribute> m_particle_attribute;  //particle attributes, used to juedge if a particle is a fluid particle

    DeviceArrayField<Real>     m_triangle_vertex_mass;  //mass of triangle vertex
    DeviceArrayField<Coord>    m_triangle_vertex;       //current positions of triangle vertexs
    DeviceArrayField<Coord>    m_triangle_vertex_old;   //positions of triangle vertexs at last time step, used to update triangle velocities
    DeviceArrayField<Triangle> m_triangle_index;        //triangle vertex's indexes

    DeviceArrayField<Coord> m_particle_force_density;  //force density of fluid particles
    DeviceArrayField<Coord> m_vertex_force_density;    //no use
    DeviceArrayField<Coord> m_vn;                      //no use

    DeviceArrayField<int> m_flip;
    Reduction<Real>*      pReduce;

    DeviceArrayField<Coord> m_velocity_mod;  //velocity norm of each particle

protected:
    bool initializeImpl() override;

private:
    int  m_pNum;
    Real m_restRho;
    int  first = 1;

    //std::shared_ptr<ConstraintModule> m_surfaceTensionSolver;
    std::shared_ptr<ConstraintModule> m_viscositySolver;  // no use

    std::shared_ptr<ConstraintModule> m_incompressibilitySolver;  // no use

    std::shared_ptr<SemiAnalyticalIncompressibilityModule<TDataType>> m_pbdModule;  //semi-analytical projection-based fluid model

    std::shared_ptr<DensityPBD<TDataType>> m_pbdModule2;  // no use

    std::shared_ptr<MeshCollision<TDataType>> m_meshCollision;  //used to handel the collision between triangles and particles

    std::shared_ptr<ImplicitViscosity<TDataType>>  m_visModule;             //viscosity
    std::shared_ptr<SurfaceTension<TDataType>>     m_surfaceTensionSolver;  //surface tension
    std::shared_ptr<Helmholtz<TDataType>>          m_Helmholtz;             //particle shifting
    std::shared_ptr<PointSetToPointSet<TDataType>> m_mapping;               //no use
    std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;            //integrator, update particle velocity and position
    std::shared_ptr<NeighborQuery<TDataType>>      m_nbrQueryPoint;         //neighbor list for particle pairs

    std::shared_ptr<NeighborQuery<TDataType>> m_nbrQueryTri;       //neighbor list for particle-triangle
    std::shared_ptr<NeighborQuery<TDataType>> m_nbrQueryTriMulti;  //abandoned
};

#ifdef PRECISION_FLOAT
template class SemiAnalyticalIncompressibleFluidModel<DataType3f>;
#else
template class SemiAnalyticalIncompressibleFluidModel<DataType3d>;
#endif
}  // namespace PhysIKA