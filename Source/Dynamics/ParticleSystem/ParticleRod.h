#pragma once
#include "ParticleSystem.h"

namespace PhysIKA
{
	template<typename> class ElasticityModule;
	template<typename> class ParticleIntegrator;
	template<typename> class FixedPoints;
	template<typename> class SimpleDamping;
	/*!
	*	\class	ParticleRod
	*	\brief	Peridynamics-based elastic object.
	*/
	template<typename TDataType>
	class ParticleRod : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleRod, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleRod(std::string name = "default");
		virtual ~ParticleRod();

		bool initialize() override;
		void advance(Real dt) override;
//		void updateTopology() override;

		void setParticles(std::vector<Coord> particles);

		void setLength(Real length);
		void setMaterialStiffness(Real stiffness);

		void setFixedParticle(int id, Coord pos);

		void getHostPosition(std::vector<Coord>& pos);

		void removeAllFixedPositions();

		bool detectCollisionPoint(int& nearParticleId, Coord preV1, Coord preV2, Coord curV1, Coord curV2);

		void doCollision(Coord pos, Coord dir);

		void setDamping(Real d);

	public:
		VarField<Real> m_horizon;

	private:
		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<ElasticityModule<TDataType>> m_elasticity;
		std::shared_ptr<FixedPoints<TDataType>> m_fixed;
		std::shared_ptr<SimpleDamping<TDataType>> m_damping;
	};


#ifdef PRECISION_FLOAT
	template class ParticleRod<DataType3f>;
#else
	template class ParticleRod<DataType3d>;
#endif
}