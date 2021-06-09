#pragma once

/**
 * @author     : Xiaosong Chen
 * @date       : 2021-05-31
 * @description: wrapper for a fast multiphase SPH solver
 * @version    : 1.0
 */


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
	*	This class wraps a fast Multiphase SPH solver
	*	
	*	Sample usage:
	* 		auto root = scene.createNewScene<FastMultiphaseSPH<DataType3f>>();
	*		root->init();
	*
	*/
	template<typename TDataType>
	class FastMultiphaseSPH : public Node
	{
		DECLARE_CLASS_1(FastMultiphaseSPH, TDataType)
	public:


		enum class particle_t { BOUDARY, FLUID, SAND };

		bool self_update = true;
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		FastMultiphaseSPH(std::string name = "default");
		virtual ~FastMultiphaseSPH();

		void initSync(); // initialize the scene

		void loadParticlesAABBVolume(Coord lo, Coord hi, Real distance, particle_t type);
		void loadParticlesAABBSurface(Coord lo, Coord hi, Real distance, particle_t type);
		void loadParticlesBallVolume(Coord center, Real r, Real distance, particle_t type);
		void loadParticlesFromFile(std::string filename, particle_t type);
		Real getSpacing() { return m_msph->h_param.spacing; }
		void setDissolutionFlag(int flag = 1) { m_msph->h_param.dissolution = flag; }


		void addParticles(const std::vector<Coord>& points, particle_t type);


		virtual void advance(Real dt) override;

		void updateTopology() override;
		bool resetStatus() override;
		void prepareData();

		// std::shared_ptr<PointRenderModule> getRenderModule();

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

		int num_o; // num of opaque particles
		DeviceArray<Vector3f> m_pos; // store all particles
		DeviceArray<Vector4f> m_color;

		DeviceArrayField<Vector3f> m_phase_concentration;


	public:
		bool initialize() override;
		//		virtual void setVisible(bool visible) override;

	protected:
		std::shared_ptr<msph::MultiphaseSPHSolver> m_msph; //!< the wrapped solver

		std::shared_ptr<PointSet<TDataType>> m_pSet;
		//std::shared_ptr<PointRenderModule> m_pointsRender;
	};


#ifdef PRECISION_FLOAT
	template class FastMultiphaseSPH<DataType3f>;
#else
	template class FastMultiphaseSPH<DataType3d>;
#endif
}