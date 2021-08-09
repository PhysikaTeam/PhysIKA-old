/**
 * @author     : He Xiaowei (xiaowei@iscas.ac.cn)
 * @date       : 2020-10-07
 * @description: Declaration of Peridynamics class, which is a container for peridynamics based deformable bodies
 * @version    : 1.0
 * 
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-06
 * @description: poslish code
 * @version    : 1.1
 * 
 */
#pragma once
#include <vector_types.h>
#include <vector>
#include "Framework/Framework/NumericalModel.h"
#include "ElasticityModule.h"
#include "Framework/Framework/FieldVar.h"

namespace PhysIKA {
template <typename>
class NeighborQuery;
template <typename>
class PointSetToPointSet;
template <typename>
class ParticleIntegrator;
template <typename>
class ElasticityModule;

/**
 * Peridynamics
 * a NumericalModel for peridynamics based deformable bodies
 * The solver is the elastic implemendataion of peridynamics
 * reference: He et al "Projective peridynamics for modeling versatile elastoplastic materials"
 *
 * Could be used by calling setNumericalModel and connecting m_position, m_velocity and m_forceDensity
 * at parent node, see ParticleCloth for example.
 * 
 * Currently seems to be used only in ParticleCloth.cpp
 *
 */

/*!
    *    \class    ParticleSystem
    *    \brief    Projective peridynamics
    *
    *    This class implements the projective peridynamics.
    *    Refer to He et al' "Projective peridynamics for modeling versatile elastoplastic materials" for details.
    */
template <typename TDataType>
class Peridynamics : public NumericalModel
{
    DECLARE_CLASS_1(Peridynamics, TDataType)

public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    Peridynamics();
    ~Peridynamics() override{};

    /*!
        *    \brief    All variables should be set appropriately before initializeImpl() is called.
        */
    bool initializeImpl() override;

    void step(Real dt) override;

public:
    VarField<Real> m_horizon;  //searching radius, default 0.0085

    DeviceArrayField<Coord> m_position;      //input and output particle positions
    DeviceArrayField<Coord> m_velocity;      //input and output particle velocities
    DeviceArrayField<Coord> m_forceDensity;  //input and output force densities on particles

    std::shared_ptr<ElasticityModule<TDataType>> m_elasticity;  //the peridynamic elastic

private:
    HostVarField<int>*  m_num;   //seems useless on current version
    HostVarField<Real>* m_mass;  //seems useless on current version

    HostVarField<Real>* m_samplingDistance;  //seems useless on current version
    HostVarField<Real>* m_restDensity;       //seems useless on current version

    std::shared_ptr<PointSetToPointSet<TDataType>> m_mapping;     //seems useless on current version
    std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;  // integrator, used to update velocities and positions
    std::shared_ptr<NeighborQuery<TDataType>>      m_nbrQuery;    //neighbor query, used to find particle neighbors
};

#ifdef PRECISION_FLOAT
template class Peridynamics<DataType3f>;
#else
template class Peridynamics<DataType3d>;
#endif
}  // namespace PhysIKA