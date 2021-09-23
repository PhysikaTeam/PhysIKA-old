#include "EmbeddedFiniteElement.h"
#include "Framework/Framework/ControllerAnimation.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Topology/PointSet.h"
#include "Core/Utility.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/NeighborQuery.h"
#include "EmbeddedIntegrator.h"
#include "FiniteElementMethod/Problem/IntegratedProblem/FEMProblemIntegratedEmbeddedElasFemProblem.h"
#include "FiniteElementMethod/Solver/FEMSolverNewtonMethod.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include <iostream>
#include <string>
#include "Core/OutputMesh.h"

using namespace std;

namespace PhysIKA {
IMPLEMENT_CLASS_1(EmbeddedFiniteElement, TDataType)

template <typename TDataType>
EmbeddedFiniteElement<TDataType>::EmbeddedFiniteElement(std::string name)
    : ParticleSystem<TDataType>(name)
{
    //m_horizon.setValue(0.0085);
    this->varHorizon()->setValue(0.0085);
    //this->attachField(&m_horizon, "horizon", "horizon");

    auto m_integrator = this->template setNumericalIntegrator<EmbeddedIntegrator<TDataType>>("integrator");
    this->currentPosition()->connect(m_integrator->inPosition());
    this->currentVelocity()->connect(m_integrator->inVelocity());
    this->currentForce()->connect(m_integrator->inForceDensity());

    /*this->getPosition()->connect2(m_integrator->m_position);
		this->getVelocity()->connect2(m_integrator->m_velocity);
		this->getForce()->connect2(m_integrator->m_forceDensity);*/

    this->getAnimationPipeline()->push_back(m_integrator);

    auto m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
    /*m_horizon.connect2(m_nbrQuery->in_Radius);
		this->current_Position.connect2(m_nbrQuery->in_Position);*/
    this->varHorizon()->connect(m_nbrQuery->inRadius());
    this->currentPosition()->connect(m_nbrQuery->inPosition());

    this->getAnimationPipeline()->push_back(m_nbrQuery);

    auto m_elasticity = this->template addConstraintModule<ElasticityModule<TDataType>>("elasticity");
    this->varHorizon()->connect(m_elasticity->inHorizon());
    this->currentPosition()->connect(m_elasticity->inPosition());
    this->currentVelocity()->connect(m_elasticity->inVelocity());
    m_nbrQuery->outNeighborhood()->connect(m_elasticity->inNeighborhood());

    /*	this->getPosition()->connect2(m_elasticity->in_Position);
		this->getVelocity()->connect2(m_elasticity->in_Velocity);
		m_horizon.connect2(m_elasticity->in_Horizon);
		m_nbrQuery->out_Neighborhood.connect2(m_elasticity->in_Neighborhood);*/

    this->getAnimationPipeline()->push_back(m_elasticity);

    //Create a node for surface mesh rendering
    m_surfaceNode = this->template createChild<Node>("Mesh");

    auto triSet = m_surfaceNode->template setTopologyModule<TriangleSet<TDataType>>("surface_mesh");

    //Set the topology mapping from PointSet to TriangleSet
    auto surfaceMapping = this->template addTopologyMapping<PointSetToPointSet<TDataType>>("surface_mapping");
    surfaceMapping->setFrom(this->m_pSet);
    surfaceMapping->setTo(triSet);
}

template <typename TDataType>
EmbeddedFiniteElement<TDataType>::~EmbeddedFiniteElement()
{
}

template <typename TDataType>
bool EmbeddedFiniteElement<TDataType>::translate(Coord t)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

    return ParticleSystem<TDataType>::translate(t);
}

template <typename TDataType>
bool EmbeddedFiniteElement<TDataType>::scale(Real s)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

    return ParticleSystem<TDataType>::scale(s);
}

template <typename TDataType>
bool EmbeddedFiniteElement<TDataType>::initialize()
{
    return ParticleSystem<TDataType>::initialize();
}

template <typename TDataType>
void EmbeddedFiniteElement<TDataType>::advance(Real dt)
{
    auto integrator = this->template getModule<EmbeddedIntegrator<TDataType>>("integrator");
    auto module     = this->template getModule<ElasticityModule<TDataType>>("elasticity");

    integrator->begin();

    integrator->integrate();

    /*	 if (module != nullptr)
		   module->constrain();*/

    integrator->end();
}

template <typename TDataType>
void EmbeddedFiniteElement<TDataType>::updateTopology()
{
    auto pts = this->m_pSet->getPoints();
    Function1Pt::copy(pts, this->currentPosition()->getValue());

    /*TODO:fix bug:
		  apply() will not update points in triSet because points in triSet has no neighbours */
    auto tMappings = this->getTopologyMappingList();
    for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
    {
        (*iter)->apply();
    }

    if (OUTPUT_MESH)
    {
        //-> output the surface mesh.
        frame_id++;
        if (frame_id % out_step != 0)
        {
            return;
        }
        auto Mesh = TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule());
        if (!Mesh)
        {
            return;
        }
        auto                                  F = Mesh->getTriangles();
        auto                                  V = Mesh->getPoints();
        std::vector<TopologyModule::Triangle> hF;
        std::vector<TDataType::Coord>         hV;
        hF.resize(F->size());
        hV.resize(V.size());
        Function1Pt::copy(hF, *F);
        Function1Pt::copy(hV, V);
        //cout << "wtf??? F.size(): " << hF.size() << endl;
        //cout << "wtf??? V.size(): " << hV.size() << endl;
        std::ofstream fout(output + std::to_string(frame_id / out_step) + ".obj");
        for (size_t i = 0; i < hV.size(); ++i)
        {
            fout << "v " << hV[i][0] << " " << hV[i][1] << " " << hV[i][2] << "\n";
        }
        for (size_t i = 0; i < hF.size(); ++i)
        {
            fout << "f " << hF[i][0] + 1 << " " << hF[i][1] + 1 << " " << hF[i][2] + 1 << "\n";
        }
        fout.close();
    }
}

template <typename TDataType>
std::shared_ptr<ElasticityModule<TDataType>> EmbeddedFiniteElement<TDataType>::getElasticitySolver()
{
    auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");
    return module;
}

template <typename TDataType>
void EmbeddedFiniteElement<TDataType>::setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver)
{
    auto nbrQuery = this->template getModule<NeighborQuery<TDataType>>("neighborhood");
    auto module   = this->template getModule<ElasticityModule<TDataType>>("elasticity");

    /*	this->getPosition()->connect2(solver->in_Position);
		this->getVelocity()->connect2(solver->in_Velocity);
		nbrQuery->out_Neighborhood.connect2(solver->in_Neighborhood);
		m_horizon.connect2(solver->in_Horizon);*/

    this->currentPosition()->connect(solver->inPosition());
    this->currentVelocity()->connect(solver->inVelocity());
    nbrQuery->outNeighborhood()->connect(solver->inNeighborhood());
    this->varHorizon()->connect(solver->inHorizon());

    this->deleteModule(module);

    solver->setName("elasticity");
    this->addConstraintModule(solver);
}

template <typename TDataType>
void EmbeddedFiniteElement<TDataType>::loadSurface(std::string filename)
{
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
}

template <typename TDataType>
std::shared_ptr<PointSetToPointSet<TDataType>> EmbeddedFiniteElement<TDataType>::getTopologyMapping()
{
    auto mapping = this->template getModule<PointSetToPointSet<TDataType>>("surface_mapping");

    return mapping;
}

template <typename TDataType>
void EmbeddedFiniteElement<TDataType>::init_problem_and_solver(const boost::property_tree::ptree& pt)
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

    epb_fac_ = std::make_shared<embedded_elas_problem_builder<Real>>(&nods[0], pt);

    auto integrator = this->template getModule<EmbeddedIntegrator<TDataType>>("integrator");
    integrator->bind_problem(epb_fac_, pt);

    auto get_file_name = [](const std::string& file) -> std::string {
        size_t pos = file.find_last_of('/') + 1;
        if (pos == std::string::npos)
        {
            pos = 0;
        }
        std::string res = file.substr(pos);
        pos             = res.find_last_of('.');
        if (pos == 0)
        {
            pos = std::string::npos;
        }
        return res.substr(0, pos);
    };
    const string filename        = get_file_name(pt.get<string>("filename"));
    const string filename_coarse = get_file_name(pt.get<string>("filename_coarse"));
    const string type            = pt.get<string>("type", "tet");
    const string type_coarse     = pt.get<string>("type_coarse", type);
    output                       = "fem_" + type + "_" + filename + "_" + type_coarse + "_" + filename_coarse + "_";
}
}  // namespace PhysIKA