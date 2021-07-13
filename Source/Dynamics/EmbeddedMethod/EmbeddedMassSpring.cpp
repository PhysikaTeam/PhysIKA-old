/**
 * @author     : ZHAO CHONGYAO (cyzhao@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: embeded mass spring interface source for physika library
 * @version    : 2.2.1
 */
#include "EmbeddedMassSpring.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Topology/PointSet.h"
#include "Core/Utility.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/NeighborQuery.h"
#include "EmbeddedIntegrator.h"
#include "Problem/integrated_problem/embedded_mass_spring_problem.h"
#include "Problem/integrated_problem/fast_ms_problem.h"
#include "Solver/newton_method.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include <iostream>

using namespace std;

namespace PhysIKA {
IMPLEMENT_CLASS_1(EmbeddedMassSpring, TDataType)

template <typename TDataType>
EmbeddedMassSpring<TDataType>::EmbeddedMassSpring(std::string name)
    : EmbeddedFiniteElement<TDataType>(name)
{
}

template <typename TDataType>
EmbeddedMassSpring<TDataType>::~EmbeddedMassSpring()
{
}

template <typename TDataType>
void EmbeddedMassSpring<TDataType>::init_problem_and_solver(const boost::property_tree::ptree& pt)
{
    auto&            m_coords = ParticleSystem<TDataType>::m_pSet->getPoints();
    HostArray<Coord> pts(m_coords.size());
    Function1Pt::copy(pts, m_coords);
    const size_t      num = pts.size();
    std::vector<Real> nods(3 * num);
#pragma omp parallel for
    for (size_t i = 0; i < num; ++i)
        for (size_t j = 0; j < 3; ++j)
            nods[j + 3 * i] = pts[i][j];

    if (pt.get<string>("solver_type") == "fast_ms")
        epb_fac_ = std::make_shared<fast_ms_builder<Real>>(&nods[0], pt);
    else
        epb_fac_ = std::make_shared<embedded_ms_problem_builder<Real>>(&nods[0], pt);

    auto integrator = this->template getModule<EmbeddedIntegrator<TDataType>>("integrator");
    integrator->bind_problem(epb_fac_, pt);
}
}  // namespace PhysIKA
