#pragma once
#include "Color.h"
#include "Vector.h"
#include "DistanceField2D.h"
using namespace std;

struct MatElement
{
	int row;
	int col;
	float  element_value;

	MatElement(int r, int c, float va)
	{
		row = r;
		col = c;
		element_value = va;
	}
};

class MeshDeformation
{
public:
	MeshDeformation(void);
	openMesh mesh;
	openMesh deformedMesh;
	openMesh constraintMesh;
	vector<openMesh::VertexHandle> constraintMeshVerHandleList;

	void UpdateConstraintMesh(int loop);

	int CreateMesh(string offfile);
	vector<openMesh::VertexHandle>  boundaryVerList;
	void CreateBoundaryList();

	//for  cloud base plane
	DistanceField2D df;
	void CreateDistance2D();
	void DrawDistance2D();


	void ClearMesh(openMesh&  mesh);
	void NormalizeMesh();
	void NormalizeMesh(openMesh& mesh);
	void ScaleMesh(openMesh& mesh, float x, float y, float z);

	//void DrawMesh();
	//void DrawDeformedMesh();
	//void DrawConstraintMesh();

	vector<MatElement> laplaceMat;
	void CreateLaplaceMatrix();
	vector<Vector3> diferentialCoordinateList;
	void CreateDifCoorList();


	int* vertexTypeList;  //0-static, 1-free, 2-anchor
	void CreateVetexType();
	void AddConstrains();


	float basePlane;

	void AddBoudaryContrainst();

	float boundary_weight;
	float interior_weight;


	int N_cons;
	void AddInteriorConstrainst();
	vector<openMesh::VertexHandle> constraintVerList;


	float diff_z_scale;
	bool UpdateMesh();

	float* noiseList;
	void CreateNoiseList();
	float* distance2Boundary;
	void CreateDis2Boundary();
	//void DrawDistance();
	void OptimizeMesh();


	void CreateOptimizedMesh(int loop);

	// front and behind height field
	float* frontHF;
	float* behindHF;
	int* pixelTypeList;
	//void DrawHeightField();     
	void ComputeTriangleNormal(float normal[3], float PA[3], float PB[3], float PC[3]);

	float dis_scale;
	float weight_dobashi;
	void CreateEntireHeightFieldAndMesh();


	//cloud data file
	int puffNumber;
	vector<Vector3>  puffPosVec;
	vector<float> puffSizeVec;
	vector<Color4> puffColorVec;
	void CloudSampling();
	void CloudSamplingSimulation(char* simulationData);
	float distanceToVolumeBoudary(float x, float y, float z);
	bool isProbabilityGreater(float threahold);
	//output cloud model: cloudxx.dat 
	void ExportCloudModel(char* cloudfile);


	void Smooth();
	void SmoothMesh(openMesh& mesh, int N);


	void MoveCurSelectedVertex(Vector3 direction, float dis);

	vector<openMesh::VertexIter> selectedVertices;
	//void SelectVertex(int x, int y);
	//void  DrawSelectedVertices();
	//Vector3 GetPixelLoc(float x, float y, float z);

	//void  Draw(DrawType dType);

	bool OutputDeformedMesh(string meshfile);  //"output.off"
	bool OutputTempMesh(char* meshfile);  //"output_temp.off"
	~MeshDeformation(void);

	struct pixelStuct
	{
		Vector3   pixel;
		openMesh::VertexIter  v_it;
		pixelStuct(Vector3 pixel, openMesh::VertexIter  v_it)
		{
			this->pixel = pixel;
			this->v_it = v_it;
		}
	};

	struct Number
	{
		int  value;
		int id;
		Number(int value, int id)
		{
			this->value = value;
			this->id = id;
		}
		bool operator <(const Number& otherNumber) const
		{
			return  this->value < otherNumber.value;
		}
	};
};

