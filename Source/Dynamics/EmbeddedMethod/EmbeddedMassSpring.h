/**
 * @author     : ZHAO CHONGYAO (cyzhao@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: embeded mass spring interface header for physika library
 * @version    : 2.2.1
 */

#pragma once
#include "ParticleSystem/ParticleSystem.h"
#include "Common/framework.h"
#include "EmbeddedIntegrator.h"
#include "EmbeddedFiniteElement.h"
#include <boost/property_tree/ptree.hpp>

namespace PhysIKA {
template <typename>
class ElasticityModule;
template <typename>
class PointSetToPointSet;

/*!
   *    \class    ExtendedEmbeddedParticleSystem
   *    \brief    Peridynamics-based elastic object.
   */
template <typename TDataType>
class EmbeddedMassSpring : public EmbeddedFiniteElement<TDataType>
{
    DECLARE_CLASS_1(ExtendedEmbeddedMassSpring, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;
    using pb_type = Problem<Real, 3>;

    EmbeddedMassSpring(std::string name = "default");
    virtual ~EmbeddedMassSpring();

    using EmbeddedFiniteElement<TDataType>::initialize;
    using EmbeddedFiniteElement<TDataType>::advance;
    using EmbeddedFiniteElement<TDataType>::updateTopology;
    using EmbeddedFiniteElement<TDataType>::translate;
    using EmbeddedFiniteElement<TDataType>::scale;
    using EmbeddedFiniteElement<TDataType>::setElasticitySolver;
    using EmbeddedFiniteElement<TDataType>::loadSurface;
    using EmbeddedFiniteElement<TDataType>::getTopologyMapping;
    using EmbeddedFiniteElement<TDataType>::getSurfaceNode;
    //        using EmbeddedFiniteElement<TDataType>::m_horizon;
    //        using EmbeddedFiniteElement<TDataType>::m_surfaceNode;

    virtual void init_problem_and_solver(const boost::property_tree::ptree& pt);

private:
    std::shared_ptr<embedded_problem_builder<Real, 3>> epb_fac_;
};

#ifdef PRECISION_FLOAT
template class EmbeddedMassSpring<DataType3f>;
#else
template class EmbeddedMassSpring<DataType3d>;
#endif
}  // namespace PhysIKA
