#include "ParticleElasticBody.h"
#include "PositionBasedFluidModel.h"

#include "Physika_Framework/Topology/TriangleSet.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Render/SurfaceMeshRender.h"
#include "Physika_Render/PointRenderModule.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Peridynamics.h"
#include "Physika_Framework/Mapping/PointSetToPointSet.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(ParticleElasticBody, TDataType)

	template<typename TDataType>
	ParticleElasticBody<TDataType>::ParticleElasticBody(std::string name)
		: ParticleSystem(name)
	{
		auto peri = std::make_shared<Peridynamics<TDataType>>();
		this->setNumericalModel(peri);
		getPosition()->connect(peri->m_position);
		getVelocity()->connect(peri->m_velocity);
		getForce()->connect(peri->m_forceDensity);

		//Create a node for surface mesh rendering
		m_surfaceNode = this->createChild<Node>("Mesh");

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		m_surfaceNode->setTopologyModule(triSet);

		auto render = std::make_shared<SurfaceMeshRender>();
		render->setColor(Vector3f(0.2f, 0.6, 1.0f));
		m_surfaceNode->addVisualModule(render);

		std::shared_ptr<PointSetToPointSet<TDataType>> surfaceMapping = std::make_shared<PointSetToPointSet<TDataType>>(m_pSet, triSet);
		this->addTopologyMapping(surfaceMapping);
	}

	template<typename TDataType>
	ParticleElasticBody<TDataType>::~ParticleElasticBody()
	{
		
	}

	template<typename TDataType>
	bool ParticleElasticBody<TDataType>::translate(Coord t)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}


	template<typename TDataType>
	bool ParticleElasticBody<TDataType>::scale(Real s)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}


	template<typename TDataType>
	bool ParticleElasticBody<TDataType>::initialize()
	{
		return ParticleSystem<TDataType>::initialize();
	}


	template<typename TDataType>
	void ParticleElasticBody<TDataType>::advance(Real dt)
	{
		auto nModel = this->getNumericalModel();
		nModel->step(this->getDt());
	}

	template<typename TDataType>
	void ParticleElasticBody<TDataType>::updateTopology()
	{
		auto pts = m_pSet->getPoints();
		Function1Pt::copy(pts, getPosition()->getValue());

		auto tMappings = this->getTopologyMappingList();
		for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
		{
			(*iter)->apply();
		}
	}


	template<typename TDataType>
	void ParticleElasticBody<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
	}


}