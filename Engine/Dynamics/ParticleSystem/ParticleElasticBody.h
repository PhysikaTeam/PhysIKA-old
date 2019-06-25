#pragma once
#include "ParticleSystem.h"

namespace Physika
{
	template<typename> class ElasticityModule;
	template<typename> class PointSetToPointSet;
	class SurfaceMeshRender;

	/*!
	*	\class	ParticleElasticBody
	*	\brief	Peridynamics-based elastic object.
	*/
	template<typename TDataType>
	class ParticleElasticBody : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleElasticBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleElasticBody(std::string name = "default");
		virtual ~ParticleElasticBody();

		bool initialize() override;
		void advance(Real dt) override;
		void updateTopology() override;

		bool translate(Coord t) override;
		bool scale(Real s) override;

		void setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver);
		std::shared_ptr<ElasticityModule<TDataType>> getElasticitySolver();
		void loadSurface(std::string filename);

		std::shared_ptr<PointSetToPointSet<TDataType>> getTopologyMapping();

		std::shared_ptr<SurfaceMeshRender> getSurfaceRender() { return m_surfaceRender; }

	public:
		VarField<Real> m_horizon;

	private:
		std::shared_ptr<Node> m_surfaceNode;
		std::shared_ptr<SurfaceMeshRender> m_surfaceRender;
	};

#ifdef PRECISION_FLOAT
	template class ParticleElasticBody<DataType3f>;
#else
	template class ParticleElasticBody<DataType3d>;
#endif
}