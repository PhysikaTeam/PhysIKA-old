#include "StaticBoundary.h"
#include "Core/Utility.h"
#include "Framework/Framework/Log.h"
#include "Framework/Framework/Node.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/ParticleSystem.h"
#include "Dynamics/ParticleSystem/BoundaryConstraint.h"

#include "Framework/Topology/DistanceField3D.h"
#include "Framework/Topology/TriangleSet.h"
#include "Rendering/SurfaceMeshRender.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(StaticBoundary, TDataType)

	template<typename TDataType>
	StaticBoundary<TDataType>::StaticBoundary()
		: Node()
	{
	}

	template<typename TDataType>
	StaticBoundary<TDataType>::~StaticBoundary()
	{
	}

	template<typename TDataType>
	bool StaticBoundary<TDataType>::addRigidBody(std::shared_ptr<RigidBody<TDataType>> child)
	{
		this->addChild(child);
		m_rigids.push_back(child);
		return true;
	}

	template<typename TDataType>
	bool StaticBoundary<TDataType>::addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child)
	{
		this->addChild(child);
		m_particleSystems.push_back(child);
		return true;
	}

	template<typename TDataType>
	void StaticBoundary<TDataType>::advance(Real dt)
	{
		for (size_t t = 0; t < m_obstacles.size(); t++)
		{

			for (int i = 0; i < m_particleSystems.size(); i++)
			{
				DeviceArrayField<Coord>* posFd = m_particleSystems[i]->getPosition();
				DeviceArrayField<Coord>* velFd = m_particleSystems[i]->getVelocity();
				m_obstacles[t]->constrain(posFd->getValue(), velFd->getValue(), dt);
			}
		}
	}

	template<typename TDataType>
	void StaticBoundary<TDataType>::loadSDF(std::string filename, bool bOutBoundary)
	{
		auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
		boundary->load(filename, bOutBoundary);

		m_obstacles.push_back(boundary);
	}


	template<typename TDataType>
	void StaticBoundary<TDataType>::loadCube(Coord lo, Coord hi, Real distance, bool bOutBoundary /*= false*/, bool bVisible)
	{
		auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
		boundary->setCube(lo, hi, distance, bOutBoundary);

		m_obstacles.push_back(boundary);

		//Note: the size of standard cube is 2m*2m*2m
		Coord scale = (hi - lo) / 2;
		Coord center = (hi + lo) / 2;

		auto m_surfaceNode = this->createChild<Node>("cube");

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		triSet->loadObjFile("../Media/standard/standard_cube.obj");
		triSet->scale(0.99*scale);
		triSet->translate(center);

		m_surfaceNode->setTopologyModule(triSet);

		auto render = std::make_shared<SurfaceMeshRender>();
		render->setColor(Vector3f(0.8, 0.8, 0.8));
		m_surfaceNode->addVisualModule(render);

		m_surfaceNode->setVisible(bVisible);
	}

	template<typename TDataType>
	void StaticBoundary<TDataType>::loadShpere(Coord center, Real r, Real distance, bool bOutBoundary /*= false*/, bool bVisible)
	{
		auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
		boundary->setSphere(center, r, distance, bOutBoundary);

		m_obstacles.push_back(boundary);

		//Note: the radius of the standard sphere is 1m
		auto m_surfaceNode = this->createChild<Node>("sphere");

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		triSet->loadObjFile("../Media/standard/standard_sphere.obj");
		triSet->scale(0.95*r);
		triSet->translate(center);

		m_surfaceNode->setTopologyModule(triSet);

		auto render = std::make_shared<SurfaceMeshRender>();
		render->setColor(Vector3f(1, 1, 0));
		m_surfaceNode->addVisualModule(render);

		m_surfaceNode->setVisible(bVisible);
	}


	template<typename TDataType>
	void StaticBoundary<TDataType>::scale(Real s)
	{
		for (int i = 0; i < m_obstacles.size(); i++)
		{
			m_obstacles[i]->m_cSDF->scale(s);
		}
	}

	template<typename TDataType>
	void StaticBoundary<TDataType>::translate(Coord t)
	{
		for (int i = 0; i < m_obstacles.size(); i++)
		{
			m_obstacles[i]->m_cSDF->translate(t);
		}
	}
}