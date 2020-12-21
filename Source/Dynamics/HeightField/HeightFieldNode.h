#pragma once
#include "Framework/Framework/Node.h"
#include "ShallowWaterEquationModel.h"
namespace PhysIKA
{
	template <typename TDataType> class HeightField;
	/*!
	*	\class	HeightField
	*	\brief	A height field node
	*/
	template<typename TDataType>
	class HeightFieldNode : public Node
	{
		DECLARE_CLASS_1(HeightFieldNode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		HeightFieldNode(std::string name = "default");
		virtual ~HeightFieldNode();


		bool initialize() override;
		void advance(Real dt) override;
		void SWEconnect();

		void loadHeightFieldParticles(Coord lo, Coord hi, int pixels, Real slope, std::vector<Coord> &vertList);
		void loadParticles(Coord lo, Coord hi, int pixels, Real slope, Real relax);
		void loadParticlesFromImage(Coord lo, Coord hi, int pixels, Real proportion, Real relax);
		//proportion :control the height

		std::vector<Real> outputDepth();
		std::vector<Real> outputSolid();
		std::vector<Real> outputUVel();
		std::vector<Real> outputWVel();

		void updateTopology() override;

	public:
		/**
		 * @brief Particle position
		 */
		DEF_EMPTY_CURRENT_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");


		/**
		 * @brief Particle velocity
		 */
		DEF_EMPTY_CURRENT_ARRAY(Velocity, Coord, DeviceType::GPU, "Particle velocity");

	private:
		Real distance;
		Real relax;
		DeviceArrayField<Real> solid;
		DeviceArrayField<Coord> normal;
		DeviceArrayField<int>  isBound;
		
		DeviceArrayField<Real> h;//water surface height
		int zcount = 0;
		int xcount = 0;


		int nx = 0;
		int nz = 0;

		std::shared_ptr<HeightField<TDataType>> m_height_field;
	};

#ifdef PRECISION_FLOAT
	template class HeightFieldNode<DataType3f>;
#else
	template class HeightFieldNode<DataType3d>;
#endif
}