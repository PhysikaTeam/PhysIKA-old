/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-08-06
 * @description: Declaration of PositionBasedFluidModelMesh class, a container for semi-analytical PBD fluids 
 *               introduced in the paper <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * @version    : 1.1
 */
#pragma once
#include "Framework/Framework/NumericalModel.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "DensityPBDMesh.h"
#include "Attribute.h"
#include "Framework/Framework/ModuleTopology.h"
#include "MeshCollision.h"


/**
 * PositionBasedFluidModelMesh
 * a NumericalModel for semi-analytical PBD fluids 
 * The solver is PBD fluids with semi-analytical boundaries
 * reference: "Semi-analytical Solid Boundary Conditions for Free Surface Flows"
 *
 * Could be used by being created and initialized at SemiAnalyticalSFINode
 * Fields required to be initialized include:
 *     m_position
 *     m_velocity
 *     m_forceDensity
 *     m_vn
 *     TriPoint
 *     TriPointOld
 *     Tri
 *     m_smoothingLength
 * 
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
class SurfaceTension;
template <typename TDataType>
class ImplicitViscosity;
template <typename TDataType>
class Helmholtz;
template <typename TDataType>
class MeshCollision;

template <typename>
class PointSetToPointSet;
typedef typename TopologyModule::Triangle Triangle;

template <typename TDataType>
class PointSet;

class ForceModule;
class ConstraintModule;
class Attribute;

template <typename TDataType>
class PositionBasedFluidModelMesh : public NumericalModel
{
    DECLARE_CLASS_1(PositionBasedFluidModelMesh, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    PositionBasedFluidModelMesh();
    virtual ~PositionBasedFluidModelMesh();

    void step(Real dt) override;

    /*
    * set smoothing lenth to be len
    */
    void setSmoothingLength(Real len)
    {
        m_smoothingLength.setValue(len);
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
    /**
    *  currently have no influence on the behaviour
    */
    void setViscositySolver(std::shared_ptr<ConstraintModule> solver);
    /**
    *  currently have no influence on the behaviour
    */
    void setSurfaceTensionSolver(std::shared_ptr<ConstraintModule> solver);


    /*
    *  have no infludence on behaviour, but can be used in visualizing densities
    */
    DeviceArrayField<Real>* getDensityField()
    {
        return &(m_pbdModule2->m_density);
    }

public:
    VarField<Real> m_smoothingLength;

    DeviceArrayField<Coord> m_position;//current particle position
    DeviceArrayField<Coord> m_velocity;//current particle velocity

    /**
    *  currently have no influence on the behaviour
    *  was used to compare the behaviour between mesh boundaries and ghost particles
    */
    DeviceArrayField<Coord> m_position_all;
    DeviceArrayField<Coord> m_position_ghost;
    DeviceArrayField<Coord> m_velocity_all;

    DeviceArrayField<Real>  m_massArray;
    DeviceArrayField<Real>  PressureFluid;
    DeviceArrayField<Real>  m_vn;
    DeviceArrayField<Coord> m_TensionForce;
    DeviceArrayField<Coord> m_forceDensity;
    DeviceArrayField<int>   ParticleId;

    DeviceArrayField<Attribute> m_attribute;
    DeviceArrayField<Coord>     m_normal;
    DeviceArrayField<int>       m_flip;

    VarField<int> Start;

    std::shared_ptr<PointSet<TDataType>> m_pSetGhost;

    DeviceArrayField<Coord>    TriPoint; //triangle vertex point position
    DeviceArrayField<Coord>    TriPointOld;//triangle vertex point position at last time step, can be used to calculate triangle velocity 
    DeviceArrayField<Triangle> Tri;//triangle index

    DeviceArrayField<Real> massTri;

protected:
    bool initializeImpl() override;

private:
    int  m_pNum;
    Real m_restRho;
    int  first = 1;

    //std::shared_ptr<ConstraintModule> m_surfaceTensionSolver;
    std::shared_ptr<ConstraintModule> m_viscositySolver;

    std::shared_ptr<ConstraintModule> m_incompressibilitySolver;

    std::shared_ptr<MeshCollision<TDataType>> m_meshCollision;

    std::shared_ptr<DensityPBDMesh<TDataType>> m_pbdModule2;

    std::shared_ptr<ImplicitViscosity<TDataType>>  m_visModule;
    std::shared_ptr<SurfaceTension<TDataType>>     m_surfaceTensionSolver;
    std::shared_ptr<Helmholtz<TDataType>>          m_Helmholtz;
    std::shared_ptr<PointSetToPointSet<TDataType>> m_mapping;
    std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
    std::shared_ptr<NeighborQuery<TDataType>>      m_nbrQueryPoint;
    std::shared_ptr<NeighborQuery<TDataType>>      m_nbrQueryPointAll;
    std::shared_ptr<NeighborQuery<TDataType>>      m_nbrQueryTri;
};

#ifdef PRECISION_FLOAT
template class PositionBasedFluidModelMesh<DataType3f>;
#else
template class PositionBasedFluidModelMesh<DataType3d>;
#endif
}  // namespace PhysIKA