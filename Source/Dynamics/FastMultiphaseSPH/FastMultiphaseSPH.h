#pragma once
#include "Framework/Framework/Node.h"
//#include "Rendering/PointRenderModule.h"
#include "solver/Multiphase/wcsph/MultiphaseSPHSolver.h"

namespace PhysIKA
{
	template <typename TDataType> class PointSet;
	/*!
	*	\class	FastMultiphaseSPH
	*	\brief	Mostly copied from ParticleSystem which is designed for Position-based fluids.
	*
	*	This class implements a fast Multiphase SPH solver
	*
	*/
	template<typename TDataType>
	class FastMultiphaseSPH : public Node
	{
		DECLARE_CLASS_1(FastMultiphaseSPH, TDataType)
	public:

		bool self_update = true;
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		FastMultiphaseSPH(std::string name = "default");
		virtual ~FastMultiphaseSPH();

		void loadParticles(Coord lo, Coord hi, Real distance);
		void loadParticles(Coord center, Real r, Real distance);
		void loadParticles(std::string filename);

		virtual bool translate(Coord t);
		virtual bool scale(Real s);

		virtual void advance(Real dt) override;

		void updateTopology() override;
		bool resetStatus() override;

		//		std::shared_ptr<PointRenderModule> getRenderModule();

				/**
				 * @brief Particle position
				 */
		DEF_EMPTY_CURRENT_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");


		/**
		 * @brief Particle velocity
		 */
		DEF_EMPTY_CURRENT_ARRAY(Velocity, Coord, DeviceType::GPU, "Particle velocity");

		/**
		 * @brief Particle force
		 */
		DEF_EMPTY_CURRENT_ARRAY(Force, Coord, DeviceType::GPU, "Force on each particle");


		DeviceArrayField<Vector3f> m_phase_concentration;


	public:
		bool initialize() override;
		//		virtual void setVisible(bool visible) override;

	protected:
		std::shared_ptr<msph::MultiphaseSPHSolver> m_msph; // float only
		std::shared_ptr<PointSet<TDataType>> m_pSet;
		//std::shared_ptr<PointRenderModule> m_pointsRender;
	};


#ifdef PRECISION_FLOAT
	template class FastMultiphaseSPH<DataType3f>;
#else
	template class FastMultiphaseSPH<DataType3d>;
#endif
}