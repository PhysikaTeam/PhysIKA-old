#include "HeightFieldNode.h"
#include "Framework/Topology/HeightField.h"
#include "ShallowWaterEquationModel.h"
#include "IO\Image_IO\image.h"
#include "IO\Image_IO\image_io.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(HeightFieldNode, TDataType)

	template<typename TDataType>
	HeightFieldNode<TDataType>::HeightFieldNode(std::string name = "default")
		: Node(name)
	{
		auto swe = this->template setNumericalModel<ShallowWaterEquationModel<TDataType>>("swe");
		this->setNumericalModel(swe);
		SWEconnect();
		
		m_height_field = std::make_shared<HeightField<TDataType>>();
		this->setTopologyModule(m_height_field);
	}

	template<typename TDataType>
	void HeightFieldNode<TDataType>::SWEconnect()
	{
		auto swe = this->getModule<ShallowWaterEquationModel<TDataType>>("swe");
		this->currentPosition()->connect(&(swe->m_position));
		
		this->currentVelocity()->connect(&(swe->m_velocity));
		this->normal.connect(&(swe->normal));

		this->isBound.connect(&(swe->isBound));
		this->solid.connect(&(swe->solid));

		swe->setDistance(distance);
		swe->setRelax(relax);
		swe->setZcount(zcount);
	}

	template<typename TDataType>
	bool HeightFieldNode<TDataType>::initialize()
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
	void HeightFieldNode<TDataType>::loadHeightFieldParticles(Coord lo, Coord hi, int pixels, Real slope, std::vector<Coord>& vertList)
	{
		//vertList返回给loadParticles，这样在初始化时能够根据solidList的情况更新
		std::vector<Coord> normalList;

		float height = 0, e = 2.71828;
		Real distance = (hi[2] - lo[2]) / (pixels - 1);
		//nx = (hi[0] - lo[0]) / distance;
		//nz = (hi[2] - lo[2]) / distance;
		//****************************************************
		float xcenter = (hi[0] - lo[0]) / 2, zcenter = (hi[2] - lo[2]) / 2;
		//float xcenter = 0.5, zcenter = 0.8;
		Real x = lo[0];
		for (int i = 0; i < pixels; i++)
		{
			Real z = lo[2];
			for (int j = 0; j < pixels; j++)
			{
				//height =  0.3 + slope * pow(e, -(pow(x - xcenter, 2) + pow(z - zcenter, 2)) * 100);
				Coord p = Coord(x, 0, z);
				vertList.push_back(Coord(x, height + lo[1], z));
				normalList.push_back(Coord(0, 1, 0));
				z += distance;
			}
			x += distance;
		}
		//标记四角
		/*int zcount = pixels;
		vertList[vertList.size()-1][1] = 1;
		vertList[vertList.size()-zcount][1] = 1;
		vertList[0][1] = 1;
		vertList[zcount-1][1] = 1;*/

		this->currentVelocity()->setElementCount(vertList.size());
		Function1Pt::copy(this->currentVelocity()->getValue(), vertList);
		velocityList.clear();
		normalList.clear();
	}

	template<typename TDataType>
	//void HeightFieldNode<TDataType>::loadParticles(Coord lo, Coord hi, Real distance,Real slope, Real relax)
	//用顶点数表示更直观一些
	void HeightFieldNode<TDataType>::loadParticles(Coord lo, Coord hi, int pixels,Real slope, Real relax)
	{
		std::vector<Coord> vertList;
		Real distance = (hi[2] - lo[2]) / (pixels-1);
		loadHeightFieldParticles(lo, hi, pixels, slope, vertList);
		this->distance = distance;
		this->relax = relax;
		std::vector<Coord> solidList;
		std::vector<Coord> normals;
		std::vector<int>  isbound;
		float height =  0, e = 2.71828;
		float xcenter = (hi[0] - lo[0]) / 2, zcenter = (hi[2] - lo[2]) / 2;
		Real x = lo[0];
		for (int i = 0; i < pixels; i++) 
		{
			Real z = lo[2];
			for (int j = 0;j < pixels; j++)
			{
				//height = 0.2+slope * pow(e, -(pow(x - xcenter, 2) + pow(z - zcenter, 2)) * 100);
				//height = z*0.45;
				if (z + distance > hi[2] || x + distance > hi[0] || x == lo[0] || z == lo[2])
					isbound.push_back(1);
				else
					isbound.push_back(0);
				//*******************判断当前position和solid哪个高
				if (lo[1] + height > vertList[j + i * pixels][1]) {
					vertList[j + i * pixels][1] = lo[1] + height;
				}
				//************************************************
				solidList.push_back(Coord(x, lo[1] + height, z));
				normals.push_back(Coord(0, 1, 0));
				z += distance;
			}
			x += distance;
		}
		xcount = pixels;
		zcount = pixels;

		nx = xcount;
		nz = zcount;

		this->currentPosition()->setElementCount(vertList.size());
		printf("%d", vertList.size());
		Function1Pt::copy(this->currentPosition()->getValue(), vertList);

		solid.setElementCount(solidList.size());
		Function1Pt::copy(solid.getValue(), solidList);

		isBound.setElementCount(solidList.size());
		Function1Pt::copy(isBound.getValue(), isbound);

		normal.setElementCount(solidList.size());
		Function1Pt::copy(normal.getValue(), normals);

		printf("distance si %f ,zcount is %d, xcount is %d\n",distance, zcount, xcount);
		vertList.clear();
		solidList.clear();
		isbound.clear();
		normals.clear();
		SWEconnect();

		this->updateTopology();

		Coord ori = Coord(0, 0, 0);
		ori[2] = -0.5 * (lo[2] + hi[2]);
		ori[0] = -0.5 * (lo[0] + hi[0]);
		m_height_field->setOrigin(ori);
	}

	template<typename TDataType>
	void HeightFieldNode<TDataType>::loadParticlesFromImage(Coord lo, Coord hi, int pixels, Real slope, Real relax)
	{
		Image *image1 = new Image;
		Image *image2 = new Image;
		//std::string filename1 = "F:\\新建文件夹\\大四第一学期\\swe\\4-4.png";//像素为1024
		//std::string filename2 = "F:\\新建文件夹\\大四第一学期\\swe\\river4-4.png";//像素为1024
		//std::string filename1 = "F:\\新建文件夹\\大四第一学期\\swe\\16-16.png";//像素为256
		//std::string filename2 = "F:\\新建文件夹\\大四第一学期\\swe\\river16-16.png";//像素为256
		std::string filename1 = "..\\..\\..\\Examples\\App_SWE\\16-16.png";//像素为256
		std::string filename2 = "..\\..\\..\\Examples\\App_SWE\\river16-16.png";//像素为256

		ImageIO::load(filename1, image1);
		ImageIO::load(filename2, image2);
		assert(image2->height() == image2->width());
		assert(image2->width() == image1->height());
		assert(image1->height() == pixels);
		assert(image1->height() == image1->width());

		std::vector<Coord> vertList;
		Real distance = (hi[2] - lo[2]) / (pixels - 1);
		loadHeightFieldParticles(lo, hi, pixels, slope, vertList);
		this->distance = distance;
		this->relax = relax;
		std::vector<Coord> solidList;
		std::vector<Coord> normals;
		std::vector<int>  isbound;
		float height = 0, e = 2.71828;
		float xcenter = (hi[0] - lo[0]) / 2, zcenter = (hi[2] - lo[2]) / 2;
		Real x = lo[0];

		for (int i = 0; i < pixels; i++)
		{
			Real z = lo[2];
			for (int j = 0; j < pixels; j++)
			{
				if (z + distance > hi[2] || x + distance > hi[0] || x == lo[0] || z == lo[2])
					isbound.push_back(1);
				else
					isbound.push_back(0);
				//*******************读入地形数据
				int temp_index = (i*pixels + j)*image1->pixelSize();
				unsigned short temp_height = (image1->rawData()[temp_index + 1] << 8) | image1->rawData()[temp_index];
				height =  temp_height * 0.3 //系数
					* (hi[1] - lo[1]) / 65535;
				//*******************************
				//*******************读入初始河流位置
				if (image2->rawData()[temp_index] == 255) {
					vertList[j + i * pixels][1] = lo[1] + height;
					height = 0;//假设河流最低点是0
				}
				solidList.push_back(Coord(x, lo[1] + height, z));
				normals.push_back(Coord(0, 1, 0));
				z += distance;
				//*******************判断当前position和solid哪个高
				if (lo[1] + height > vertList[j + i * pixels][1]) {
					vertList[j + i * pixels][1] = solidList[j + i * pixels][1];
				}
				//***********************************************
			}
			x += distance;
		}
		xcount = pixels;
		zcount = pixels;

		nx = xcount;
		nz = zcount;

		this->currentPosition()->setElementCount(vertList.size());
		printf("%d", vertList.size());
		Function1Pt::copy(this->currentPosition()->getValue(), vertList);

		solid.setElementCount(solidList.size());
		Function1Pt::copy(solid.getValue(), solidList);

		isBound.setElementCount(solidList.size());
		Function1Pt::copy(isBound.getValue(), isbound);

		normal.setElementCount(solidList.size());
		Function1Pt::copy(normal.getValue(), normals);

		printf("distance si %f ,zcount is %d, xcount is %d\n", distance, zcount, xcount);
		vertList.clear();
		solidList.clear();
		isbound.clear();
		normals.clear();
		//delete[] image1;做不到，因为外界无法访问（删除）他的私有元素，其实也不需要做
		SWEconnect();

		this->updateTopology();

		Coord ori = Coord(0,0,0);
		ori[2] = -0.5 * (lo[2] + hi[2]);
		ori[0] = -0.5 * (lo[0] + hi[0]);
		m_height_field->setOrigin(ori);

	}
	template<typename TDataType>
	HeightFieldNode<TDataType>::~HeightFieldNode()
	{
	}
	template<typename TDataType>
	void HeightFieldNode<TDataType>::advance(Real dt)
	{
		auto nModel = this->getNumericalModel();
		nModel->step(dt);
	}

	template<typename Real, typename Coord>
	__global__ void SetupHeights(
		DeviceArray2D<Real> height, 
		DeviceArray<Coord> pts)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		if (i < height.Nx() && j < height.Ny())
		{
			int id = j + i * (height.Nx());
			height(i, j) = pts[id][1];
		}
	}

	template<typename TDataType>
	void HeightFieldNode<TDataType>::updateTopology()
	{
		if (!this->currentPosition()->isEmpty())
		{
			int num = this->currentPosition()->getElementCount();
			auto& pts = this->currentPosition()->getValue();

			m_height_field->setSpace(distance,distance);
			//m_height_field->setSpace(0.005, 0.005);
			auto& heights = m_height_field->getHeights();
			printf("heights.Nx and heights.Nz is %d,%d\n", heights.Nx(), heights.Ny());
			if (nx != heights.Nx() || nz != heights.Ny())
			{
				heights.resize(nx, nz);
				printf("heights.Nx and heights.Nz is %d,%d\n", heights.Nx(), heights.Ny());
			}

			uint3 total_size;
			total_size.x = nx;
			total_size.y = nz;
			total_size.z = 1;

			//ti++;

			cuExecute3D(total_size, SetupHeights,
				heights,
				pts);
		}
	}

}