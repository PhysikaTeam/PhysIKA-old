#include "FastMultiphaseSPH.h"

#include "Framework/Topology/PointSet.h"
#include "Core/Utility.h"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(FastMultiphaseSPH, TDataType)

		template<typename TDataType>
	FastMultiphaseSPH<TDataType>::FastMultiphaseSPH(std::string name)
		: Node(name)
	{
		//		attachField(&m_velocity, MechanicalState::velocity(), "Storing the particle velocities!", false);
		//		attachField(&m_force, MechanicalState::force(), "Storing the force densities!", false);

		m_pSet = std::make_shared<PointSet<TDataType>>();
		this->setTopologyModule(m_pSet);

		m_msph = std::make_shared<msph::MultiphaseSPHSolver>();
		m_msph->init();
		
		prepareData();
		
		std::vector<Coord> buffer(num_o);
		m_pSet->setPoints(buffer);
		m_pSet->setNormals(buffer);
		m_phase_concentration.setElementCount(num_o);

		updateTopology();
		// 		m_pointsRender = std::make_shared<PointRenderModule>();
		// 		this->addVisualModule(m_pointsRender);
	}
	struct OpaquePred {
		__host__ __device__
		bool operator()(Vector4f v) {
			return v[3] != 0;
		}
	};
	template<typename TDataType>
	void FastMultiphaseSPH<TDataType>::prepareData()
	{
		// get all particles
		int num = m_msph->num_particles;
		if (num != m_pos.size()) {
			m_pos.resize(num);
			m_color.resize(num);
		}
		m_msph->prepareRenderData((cfloat3*)m_pos.getDataPtr(), (cfloat4*)m_color.getDataPtr());
		num_o = num;
		// then filter transparent particles
		Vector3f* d_pos = m_pos.getDataPtr();
		Vector4f* d_color = m_color.getDataPtr();
		thrust::copy_if(thrust::device, d_pos, d_pos + num, d_color, d_pos, OpaquePred());
		auto oe = thrust::copy_if(thrust::device, d_color, d_color + num, d_color, OpaquePred());
		num_o = oe - d_color;
	}

	template<typename TDataType>
	FastMultiphaseSPH<TDataType>::~FastMultiphaseSPH()
	{

	}

	template<typename TDataType>
	void FastMultiphaseSPH<TDataType>::advance(Real dt)
	{
		// dt not used here as its managed by external solver ...
		m_msph->step();
	}


	template<typename TDataType>
	void FastMultiphaseSPH<TDataType>::loadParticles(std::string filename)
	{
		m_pSet->loadObjFile(filename);
	}

	template<typename TDataType>
	void FastMultiphaseSPH<TDataType>::loadParticles(Coord center, Real r, Real distance)
	{
		std::vector<Coord> vertList;
		std::vector<Coord> normalList;

		Coord lo = center - r;
		Coord hi = center + r;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p = Coord(x, y, z);
					if ((p - center).norm() < r)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}
		normalList.resize(vertList.size());

		m_pSet->setPoints(vertList);
		m_pSet->setNormals(normalList);

		vertList.clear();
		normalList.clear();
	}

	template<typename TDataType>
	void FastMultiphaseSPH<TDataType>::loadParticles(Coord lo, Coord hi, Real distance)
	{
		std::vector<Coord> vertList;
		std::vector<Coord> normalList;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p = Coord(x, y, z);
					vertList.push_back(Coord(x, y, z));
				}
			}
		}
		normalList.resize(vertList.size());

		m_pSet->setPoints(vertList);
		m_pSet->setNormals(normalList);

		std::cout << "particle number: " << vertList.size() << std::endl;

		vertList.clear();
		normalList.clear();
	}

	template<typename TDataType>
	bool FastMultiphaseSPH<TDataType>::translate(Coord t)
	{
		m_pSet->translate(t);

		return true;
	}


	template<typename TDataType>
	bool FastMultiphaseSPH<TDataType>::scale(Real s)
	{
		m_pSet->scale(s);

		return true;
	}

	template<typename TDataType>
	bool FastMultiphaseSPH<TDataType>::initialize()
	{
		return Node::initialize();
	}

	// 	template<typename TDataType>
	// 	void FastMultiphaseSPH<TDataType>::setVisible(bool visible)
	// 	{
	// 		if (m_pointsRender == nullptr)
	// 		{
	// 			m_pointsRender = std::make_shared<PointRenderModule>();
	// 			this->addVisualModule(m_pointsRender);
	// 		}
	// 
	// 		Node::setVisible(visible);
	// 	}

	struct ColorOp {
		__host__ __device__
			Vector3f operator()(Vector4f color) {
			return Vector3f(1 - color[0], 1 - color[0], 1 - color[0]);
		}
	};

	template<typename TDataType>
	void FastMultiphaseSPH<TDataType>::updateTopology()
	{
		//if (!this->currentPosition()->isEmpty())
		//{
		//	int num = this->currentPosition()->getElementCount();
		//	auto& pts = m_pSet->getPoints();
		//	if (num != pts.size())
		//	{
		//		pts.resize(num);
		//	}
		//	Function1Pt::copy(pts, this->currentPosition()->getValue());
		//}
		prepareData();
		auto pts = m_pSet->getPoints();
		cudaMemcpy(pts.getDataPtr(), m_pos.getDataPtr(), sizeof(Coord) * num_o, cudaMemcpyDeviceToDevice);
		Vector3f* color_idx = m_phase_concentration.getValue().getDataPtr();
		thrust::transform(thrust::device, m_color.getDataPtr(), m_color.getDataPtr() + num_o, color_idx, ColorOp());
	}


	template<typename TDataType>
	bool FastMultiphaseSPH<TDataType>::resetStatus()
	{
		auto pts = m_pSet->getPoints();

		if (pts.size() > 0)
		{
			this->currentPosition()->setElementCount(pts.size());
			this->currentVelocity()->setElementCount(pts.size());
			this->currentForce()->setElementCount(pts.size());

			Function1Pt::copy(this->currentPosition()->getValue(), pts);
			this->currentVelocity()->getReference()->reset();
		}

		return Node::resetStatus();
	}

	// 	template<typename TDataType>
	// 	std::shared_ptr<PointRenderModule> FastMultiphaseSPH<TDataType>::getRenderModule()
	// 	{
	// // 		if (m_pointsRender == nullptr)
	// // 		{
	// // 			m_pointsRender = std::make_shared<PointRenderModule>();
	// // 			this->addVisualModule(m_pointsRender);
	// // 		}
	// 
	// 		return m_pointsRender;
	// 	}
}