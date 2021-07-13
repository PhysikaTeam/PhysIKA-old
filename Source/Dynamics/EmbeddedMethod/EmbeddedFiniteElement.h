/**
 * @author     : ZHAO CHONGYAO (cyzhao@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: Finite Element method header for physika library
 * @version    : 2.2.1
 */
#pragma once
#include "ParticleSystem/ParticleSystem.h"
#include "Common/framework.h"
#include "EmbeddedIntegrator.h"
#include <boost/property_tree/ptree.hpp>

namespace PhysIKA {
template <typename>
class ElasticityModule;
template <typename>
class PointSetToPointSet;

/*!
   *    \class    EmbeddedFiniteElement
   *    \brief    Peridynamics-based elastic object.
   */
template <typename TDataType>
class EmbeddedFiniteElement : public ParticleSystem<TDataType>
{
    DECLARE_CLASS_1(EmbeddedFiniteElement, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;
    using pb_type = Problem<Real, 3>;

    EmbeddedFiniteElement(std::string name = "default");
    virtual ~EmbeddedFiniteElement();

    bool initialize() override;
    void advance(Real dt) override;
    void updateTopology() override;

    bool translate(Coord t) override;
    bool scale(Real s) override;

    void                                         setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver);
    std::shared_ptr<ElasticityModule<TDataType>> getElasticitySolver();
    void                                         loadSurface(std::string filename);

    std::shared_ptr<PointSetToPointSet<TDataType>> getTopologyMapping();

    std::shared_ptr<Node> getSurfaceNode()
    {
        return m_surfaceNode;
    }
    virtual void init_problem_and_solver(const boost::property_tree::ptree& pt);

public:
    /*VarField<Real> m_horizon;*/
    DEF_EMPTY_VAR(Horizon, Real, "Horizon");

protected:
    std::shared_ptr<Node>                                m_surfaceNode;
    std::shared_ptr<embedded_elas_problem_builder<Real>> epb_fac;
};

#ifdef PRECISION_FLOAT
template class EmbeddedFiniteElement<DataType3f>;
#else
template class EmbeddedFiniteElement<DataType3d>;
#endif
}  // namespace PhysIKA
