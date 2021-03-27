#include "ParticleCloth.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"
#include "Core/Utility.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Dynamics/ParticleSystem/Peridynamics.h"
#include "Dynamics/ParticleSystem/FixedPoints.h"

#include "IO/Surface_Mesh_IO/ObjFileLoader.h"

namespace PhysIKA
{
	int objFileNum = 1;

	IMPLEMENT_CLASS_1(ParticleCloth, TDataType)

	template<typename TDataType>
	ParticleCloth<TDataType>::ParticleCloth(std::string path, std::string name)
		: path(path), ParticleSystem<TDataType>(name)
	{
		auto peri = std::make_shared<Peridynamics<TDataType>>();
		this->setNumericalModel(peri);
		this->currentPosition()->connect(&peri->m_position);
		this->currentVelocity()->connect(&peri->m_velocity);
		this->currentForce()->connect(&peri->m_forceDensity);

		auto fixed = std::make_shared<FixedPoints<TDataType>>();


		//Create a node for surface mesh rendering
		m_surfaceNode = this->template createChild<Node>("Mesh");

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		m_surfaceNode->setTopologyModule(triSet);

		//auto render = std::make_shared<SurfaceMeshRender>();
		//render->setColor(Vector3f(0.4, 0.75, 1));
		//m_surfaceNode->addVisualModule(render);

		std::shared_ptr<PointSetToPointSet<TDataType>> surfaceMapping = std::make_shared<PointSetToPointSet<TDataType>>(this->m_pSet, triSet);
		this->addTopologyMapping(surfaceMapping);

		//this->setVisible(true);
	}

	template<typename TDataType>
	ParticleCloth<TDataType>::~ParticleCloth()
	{
		
	}

	template<typename TDataType>
	bool ParticleCloth<TDataType>::translate(Coord t)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}


	template<typename TDataType>
	bool ParticleCloth<TDataType>::scale(Real s)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}


	template<typename TDataType>
	bool ParticleCloth<TDataType>::initialize()
	{
		return ParticleSystem<TDataType>::initialize();
	}


	template<typename TDataType>
	void ParticleCloth<TDataType>::advance(Real dt)
	{
		auto nModel = this->getNumericalModel();
		nModel->step(this->getDt());
	}

	template<typename TDataType>
	void ParticleCloth<TDataType>::updateTopology()
	{
		std::string fileName = path + "cloth_"+std::to_string(objFileNum-1) + ".obj";
		objFileNum++;

		std::cout << "fileName :" << fileName << std::endl;

		auto pts = this->m_pSet->getPoints();
		Function1Pt::copy(pts, this->currentPosition()->getValue());

 		auto tMappings = this->getTopologyMappingList();
		for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
		{
			(*iter)->apply();
			auto tptr = std::static_pointer_cast<PointSetToPointSet<TDataType>>(*iter);

			auto tpSet = tptr->getTo();
			auto triSet = std::static_pointer_cast<TriangleSet<TDataType>>(tpSet);
			auto triangles = triSet->getTriangles();
			auto trisHost = triangles->CopyDeviceToHost();
			int triNums = triangles->size();

			auto pSet = tptr->getFrom();
			auto points = pSet->getPoints();
			auto pointsHost = points.CopyDeviceToHost();
			int pointNums = points.size();

			if (path != "" &&  objFileNum >= 200)
			{
				std::string fileName = path + "cloth_" + std::to_string(objFileNum - 200) + ".obj";
				ObjFileLoader::save(fileName, trisHost, triNums, pointsHost, pointNums);
			}
				
		}
	}

	template<typename TDataType>
	void ParticleCloth<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::CastPointerDown<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
	}
}