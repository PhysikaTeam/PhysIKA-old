#include "HeightField.h"
#include "PositionBasedFluidModel.h"
#include "ShallowWaterEquationModel.h"
namespace PhysIKA
{
	IMPLEMENT_CLASS_1(HeightField, TDataType)

	template<typename TDataType>
	HeightField<TDataType>::HeightField(std::string name = "default")
		: ParticleSystem<TDataType>(name)
	{
		auto swe = this->template setNumericalModel<ShallowWaterEquationModel<TDataType>>("swe");
		this->setNumericalModel(swe);
		SWEconnect();
		
	}

	template<typename TDataType>
	void HeightField<TDataType>::SWEconnect()
	{
		auto swe = this->getModule<ShallowWaterEquationModel<TDataType>>("swe");
		//auto swe = this->template setNumericalModel<ShallowWaterEquationModel<TDataType>>("swe2");
		//this->setNumericalModel(swe);
		//std::shared_ptr<ShallowWaterEquationModel<TDataType>> swe = this->getNumericalModel();
		this->currentPosition()->connect(&(swe->m_position));
		
		this->currentVelocity()->connect(&(swe->m_velocity));
		//this->h.connect(swe->h);
		this->normal.connect(&(swe->normal));

		this->neighbors.connect(&(swe->neighborIndex));
		this->isBound.connect(&(swe->isBound));
		this->solid.connect(&(swe->solid));

		swe->setDistance(distance);
		swe->setRelax(relax);
	}

	template<typename TDataType>
	bool HeightField<TDataType>::initialize()
	{
		return Node::initialize();
	}

	//template<typename Real, typename Coord>
	__global__ void InitNeighbor(
		NeighborList<int> neighbors,
		int zcount,
		int xcount)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= neighbors.size()) return;
		if(i%zcount==0)
			neighbors.setElement(i, 0, - 1);
		else
			neighbors.setElement(i, 0, i - 1);
		if((i+1)%zcount == 0)
			neighbors.setElement(i, 1, -1);
		else
			neighbors.setElement(i, 1, i + 1);

		neighbors.setElement(i, 2, i - zcount);
		neighbors.setElement(i, 3, i + zcount);

	}

	template<typename TDataType>
	void HeightField<TDataType>::loadParticles(Coord lo, Coord hi, Real distance,Real slope, Real relax)
	{
		loadHeightFieldParticles(lo, hi, distance, slope);
		
		this->distance = distance;
		this->relax = relax;
		std::vector<Coord> solidList;
		std::vector<Coord> normals;
		std::vector<int>  isbound;
		float height;
		int xcount = 0;
		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			xcount++;
			for (Real z = lo[2]; z <= hi[2]; z += distance)
			{
				height = 20 * (x + z - lo[0] - lo[2]);
				if (z + distance > hi[2] || x + distance > hi[0] || x == lo[0] || z == lo[2])
					isbound.push_back(1);
				else
					isbound.push_back(0);
				//	height = 
				solidList.push_back(Coord(x, lo[1], z));
				//wh.push_back(height - lo[1]);
				normals.push_back(Coord(0, 1, 0));
			}
		}

		solid.setElementCount(solidList.size());
		Function1Pt::copy(solid.getValue(), solidList);

		//h.setElementCount(solidList.size());
		//Function1Pt::copy(h.getValue(), wh);

		isBound.setElementCount(solidList.size());
		Function1Pt::copy(isBound.getValue(), isbound);

		normal.setElementCount(solidList.size());
		Function1Pt::copy(normal.getValue(), normals);
		//m_velocity.setElementCount(solidList.size());
		//neighbors.resize(solidList.size(), 4);
		neighbors.setElementCount(solidList.size(), 4);
		//add four neighbors:up down left right
		int zcount = solidList.size() / xcount;
		int num = solidList.size();
		printf("zcount is %d, xcount is %d\n", zcount, xcount);
		cuint pDims = cudaGridSize(num, BLOCK_SIZE);
		InitNeighbor << < pDims, BLOCK_SIZE >> > (neighbors.getValue(), zcount, xcount);
		cuSynchronize();
		solidList.clear();
		
		isbound.clear();
		normals.clear();

		DeviceArrayField<Coord> pos = *(this->currentPosition());
		SWEconnect();
	}

	template<typename TDataType>
	HeightField<TDataType>::~HeightField()
	{
	}
	template<typename TDataType>
	void HeightField<TDataType>::advance(Real dt)
	{
		// 		auto pbf = this->getModule<PositionBasedFluidModel<TDataType>>("pbd");
		// 
		// 		pbf->getDensityField()->connect(this->getRenderModule()->m_scalarIndex);
		// 		this->getRenderModule()->setColorRange(950, 1100);
		// 		this->getRenderModule()->setReferenceColor(1000);

		auto nModel = this->getNumericalModel();
		nModel->step(dt);
	}
	//template<typename TDataType>
	//void HeightField<TDataType>::loadObjFile(std::string filename)
	//{
	//	
	//}

}