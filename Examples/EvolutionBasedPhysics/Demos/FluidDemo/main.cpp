#include "Common/Common.h"
#include "GL/glew.h"
#include "Demos/Visualization/MiniGL.h"
#include "Demos/Visualization/Selection.h"
#include "GL/glut.h"
#include "Demos/Simulation/TimeManager.h"
#include <Eigen/Dense>
#include "FluidModel.h"
#include "TimeStepFluidModel.h"
#include <iostream>
#include "Demos/Utils/Logger.h"
#include "Demos/Utils/Timing.h"
#include "Demos/Utils/FileSystem.h"
#include "main.h"

#include <random>

#define _USE_MATH_DEFINES
#include "math.h"


#define SDF_FILE "clouds\\cloud0.cdf"

// Enable memory leak detection
#if defined(_DEBUG) && !defined(EIGEN_ALIGN)
	#define new DEBUG_NEW 
#endif

INIT_TIMING
INIT_LOGGING

//# define USING_GUI 1

using namespace PBD;
using namespace Eigen;
using namespace std;
using namespace Utilities;

void timeStep (std::string& rootPath);
void buildModel ();
void createBreakingDam();
void addWall(const Vector3r &minX, const Vector3r &maxX, std::vector<Vector3r> &boundaryParticles);
void initBoundaryData(std::vector<Vector3r> &boundaryParticles);
void render ();
void cleanup();
void reset();
void selection(const Eigen::Vector2i &start, const Eigen::Vector2i &end);
void createSphereBuffers(Real radius, int resolution);
void renderSphere(const Vector3r &x, const float color[]);
void releaseSphereBuffers();
void TW_CALL setTimeStep(const void *value, void *clientData);
void TW_CALL getTimeStep(void *value, void *clientData);
void TW_CALL setVelocityUpdateMethod(const void *value, void *clientData);
void TW_CALL getVelocityUpdateMethod(void *value, void *clientData);
void TW_CALL setViscosity(const void *value, void *clientData);
void TW_CALL getViscosity(void *value, void *clientData);

//zzl ���ӵĺ���
//���ļ�����������
void addParticleFromFile(std::string &fileName, std::vector<Vector3r> &fluidParticles);
void generateParticleForModel();
void matchControlAndTargetParticles(vector<Vector3r> &iniShapeParticle, vector<Vector3r>&tarShapeParticle, vector<unsigned int> &controlParticleIndex, vector<unsigned int> &targetParticleIndex);
void generateControlAndTargetParticle(vector<Vector3r> &iniShapeParticle, vector<Vector3r>&tarShapeParticle, vector<unsigned int> &controlParticleIndex, vector<unsigned int> &targetParticleIndex);
void addParticleForControl(std::vector<Vector3r> &fluidParticles, unsigned int flag);



FluidModel model;
TimeStepFluidModel simulation;
SDFGradientField sdfGraField; //����ȫ�ֱ���sdfGraField�洢�����Ƴ�
//����ȫ�ֵ�SDF
//std::string cdfName = "clouds\\bunny_10k.cdf";
//std::string iniShapeName = "clouds\\cloud69.dat";

//-------------------zzl-------2019-1-13-----------start---------------------

//generate data for learning
//std::string cdfName = "cloud4learning\\tran-scale-bunny.cdf"; //Ŀ����״�ķ��ž��볡
//std::string tarShapeName = "cloud4learning\\tran-scale-bunny.dat"; //Ŀ����״����������
//
//std::string iniShapeName = "cloud4learning\\cloud64-high.dat"; //Դ��״����������
//std::string oriCdfName = "cloud4learning\\cloud64.cdf"; //Դ��״�ķ��ž��볡

//----------------------69-bunny---------------------------------------------------
//std::string cdfName = "cloud4learning\\tran-scale-bunny.cdf"; //Ŀ����״�ķ��ž��볡
//std::string tarShapeName = "cloud4learning\\tran-scale-bunny.dat"; //Ŀ����״����������
//
//std::string iniShapeName = "cloud4learning\\cloud69.dat"; //Դ��״����������
//std::string oriCdfName = "cloud4learning\\cloud69.cdf"; //Դ��״�ķ��ž��볡


//----------------------69-64---------------------------------------------------
//std::string cdfName = "cloud4learning\\cloud64.cdf"; //Ŀ����״�ķ��ž��볡
//std::string tarShapeName = "cloud4learning\\cloud64.dat"; //Ŀ����״����������
//
//std::string iniShapeName = "cloud4learning\\cloud69.dat"; //Դ��״����������
//std::string oriCdfName = "cloud4learning\\cloud69.cdf"; //Դ��״�ķ��ž��볡

//----------------------dragon-bunny---------------------------------------------------
std::string cdfName = "D:\\Code\\PositionBasedDynamics-master\\cloud4learning\\tran-scale-bunny.cdf"; //Ŀ����״�ķ��ž��볡
std::string tarShapeName = "D:\\Code\\PositionBasedDynamics-master\\cloud4learning\\tran-scale-bunny.dat"; //Ŀ����״����������

std::string iniShapeName = "D:\\Code\\PositionBasedDynamics-master\\cloud4learning\\tran-scale-dragon-new.dat"; //Դ��״����������
std::string oriCdfName = "D:\\Code\\PositionBasedDynamics-master\\cloud4learning\\tran-scale-dragon.cdf"; //Դ��״�ķ��ž��볡


Discregrid::CubicLagrangeDiscreteGrid sdf = simulation.computeSDF(cdfName);
Discregrid::CubicLagrangeDiscreteGrid oriSdf = simulation.computeSDF(oriCdfName);
unsigned int ConstParticleNum = 10800; //ָ����������е�������

unsigned int oriSurfaceParticleNum = 0; //ԴĿ����״����������
unsigned int oriParticleNum = 0; //ԴĿ����������
unsigned int tarSurfaceParticleNum = 0; //Ŀ����״����������
unsigned int tarParticleNum = 0; //Ŀ����״������

//-------------------zzl-------2019-1-13-----------end---------------------


const Real particleRadius = 0.02;//0.025;
const unsigned int width = 15;//15;
const unsigned int depth = 15;//15;
const unsigned int height = 20;// 20;
const Real containerWidth = (width + 1)*particleRadius*2.0 * 5.0;//*10;
const Real containerDepth = (depth + 1)*particleRadius*2.0;// *10;
const Real containerHeight = 4.0;//*10;

//������沽��
int timeSteps = 0;

//������滷���ռ�������С
const int xHalfGridNum = 80;
const int yHalfGridNum = 80;
const int zHalfGridNum = 80;

bool doPause = false;
std::vector<unsigned int> selectedParticles;
Vector3r oldMousePos;
// initiate buffers
GLuint elementbuffer;
GLuint normalbuffer;
GLuint vertexbuffer;
int vertexBufferSize = 0;
GLint context_major_version, context_minor_version;
string exePath;
string dataPath;


int fluidEvaluation(std::string& oriCdfName, 
					std::string& oriShapeName, 
					std::string& tarCdfName, 
					std::string& tarShapeName,
					std::string& rootPath,
					int max_steps)
{
	REPORT_MEMORY_LEAKS

	::oriCdfName = oriCdfName;
	iniShapeName = oriShapeName;
	cdfName = tarCdfName;
	::tarShapeName = tarShapeName;

	oriSdf = simulation.computeSDF(oriCdfName);
	sdf = simulation.computeSDF(tarCdfName);


	std::string logPath = FileSystem::normalizePath(FileSystem::getProgramPath() + "/log");
	FileSystem::makeDirs(logPath);
	logger.addSink(unique_ptr<ConsoleSink>(new ConsoleSink(LogLevel::INFO)));
	logger.addSink(unique_ptr<FileSink>(new FileSink(LogLevel::DEBUG, logPath + "/PBD.log")));

	buildModel();
	if (context_major_version >= 3)
		createSphereBuffers((Real)particleRadius, 8);

	for (int j = 0; j < max_steps; j++)
	{
		timeStep(rootPath);
	}

	cleanup();
	Timing::printAverageTimes();

	return 1;
}

// main 
int main( int argc, char **argv )
{
	fluidEvaluation(std::string("D:\\Code\\PositionBasedDynamics-master\\cloud4learning\\tran-scale-horse.cdf"),
		std::string("D:\\Code\\PositionBasedDynamics-master\\cloud4learning\\tran-scale-horse.dat"),
		std::string("D:\\Code\\PositionBasedDynamics-master\\cloud4learning\\tran-scale-person.cdf"),
		std::string("D:\\Code\\PositionBasedDynamics-master\\cloud4learning\\tran-scale-person.dat"),
		std::string("D:\\Code\\PositionBasedDynamics-master\\CloudResults\\"),
		100
		);
	//��������
	//generateParticleForModel();

	REPORT_MEMORY_LEAKS

	std::string logPath = FileSystem::normalizePath(FileSystem::getProgramPath() + "/log");
	FileSystem::makeDirs(logPath);
	logger.addSink(unique_ptr<ConsoleSink>(new ConsoleSink(LogLevel::INFO)));
	logger.addSink(unique_ptr<FileSink>(new FileSink(LogLevel::DEBUG, logPath + "/PBD.log")));

	//exePath = FileSystem::getProgramPath();
	//dataPath = exePath + "/" + std::string(PBD_DATA_PATH);

#ifdef USING_GUI
	//OpenGL
	MiniGL::init (argc, argv, 1024, 768, 0, 0, "Fluid demo");
	MiniGL::initLights ();
	MiniGL::setClientIdleFunc (50, timeStep);		
	MiniGL::setKeyFunc(0, 'r', reset);
	MiniGL::setSelectionFunc(selection);

	MiniGL::getOpenGLVersion(context_major_version, context_minor_version);

	MiniGL::setClientSceneFunc(render);			
	MiniGL::setViewport (40.0, 0.1f, 500.0, Vector3r (0.0, 3.0, 8.0), Vector3r (0.0, 0.0, 0.0));

	TwAddVarRW(MiniGL::getTweakBar(), "Pause", TW_TYPE_BOOLCPP, &doPause, " label='Pause' group=Simulation key=SPACE ");
	TwAddVarCB(MiniGL::getTweakBar(), "TimeStepSize", TW_TYPE_REAL, setTimeStep, getTimeStep, &model, " label='Time step size'  min=0.0 max = 0.1 step=0.001 precision=4 group=Simulation ");
	TwType enumType = TwDefineEnum("VelocityUpdateMethodType", NULL, 0);
	TwAddVarCB(MiniGL::getTweakBar(), "VelocityUpdateMethod", enumType, setVelocityUpdateMethod, getVelocityUpdateMethod, &simulation, " label='Velocity update method' enum='0 {First Order Update}, 1 {Second Order Update}' group=Simulation");
	TwAddVarCB(MiniGL::getTweakBar(), "Viscosity", TW_TYPE_REAL, setViscosity, getViscosity, &model, " label='Viscosity'  min=0.0 max = 0.5 step=0.001 precision=4 group=Simulation ");
#endif // USING_GUI

	buildModel();

	if (context_major_version >= 3)
		createSphereBuffers((Real)particleRadius, 8);

#ifdef USING_GUI
	glutMainLoop ();
#else
	for (int j = 0; j < 100; j++)
	{
		timeStep(std::string("D:\\Code\\PositionBasedDynamics-master\\CloudResults\\"));
	}
#endif // USING_GUI

	cleanup();

	Timing::printAverageTimes();
	
	return 0;
}

void cleanup()
{
	delete TimeManager::getCurrent();
	if (context_major_version >= 3)
		releaseSphereBuffers();
}

void reset()
{
	Timing::printAverageTimes();
	Timing::reset();

	model.reset();
	simulation.reset();
	TimeManager::getCurrent()->setTime(0.0);
}

void mouseMove(int x, int y)
{
	Vector3r mousePos;
	MiniGL::unproject(x, y, mousePos);
	const Vector3r diff = mousePos - oldMousePos;

	TimeManager *tm = TimeManager::getCurrent();
	const Real h = tm->getTimeStepSize();

	ParticleData &pd = model.getParticles();
	for (unsigned int j = 0; j < selectedParticles.size(); j++)
	{
		pd.getVelocity(selectedParticles[j]) += 5.0*diff/h;
	}
	oldMousePos = mousePos;
}

void selection(const Eigen::Vector2i &start, const Eigen::Vector2i &end)
{
	std::vector<unsigned int> hits;
	selectedParticles.clear();
	ParticleData &pd = model.getParticles();
	Selection::selectRect(start, end, &pd.getPosition(0), &pd.getPosition(pd.size() - 1), selectedParticles);
	if (selectedParticles.size() > 0)
		MiniGL::setMouseMoveFunc(GLUT_MIDDLE_BUTTON, mouseMove);
	else
		MiniGL::setMouseMoveFunc(-1, NULL);

	MiniGL::unproject(end[0], end[1], oldMousePos);
}


void timeStep(std::string& rootPath)
{
	if (doPause)
		return;
	//-----------abstract the file name from the given full file name----
	std::string iniName;
	std::string tarName;

	int iniNameBegin, iniNameEnd;
	int tarNameBegin, tarNameEnd;
	iniNameBegin =  iniShapeName.find_last_of('\\') + 1;
	iniNameEnd = iniShapeName.find_last_of('.') - 1;
	iniName = iniShapeName.substr(iniNameBegin, iniNameEnd - iniNameBegin + 1);
	tarNameBegin = cdfName.find_last_of('\\') + 1;
	tarNameEnd = cdfName.find_last_of('.') - 1;
	tarName = cdfName.substr(tarNameBegin, tarNameEnd - tarNameBegin + 1);
	//-------------------------------------------------------------------
	// Simulation code
	timeSteps++;
	//std::cout << "timeStep=" << timeSteps << std::endl;
	for (unsigned int i = 0; i < 1; i++)
		simulation.step(rootPath, model,sdfGraField,sdf,iniName,tarName);
}


void buildModel ()
{
	TimeManager::getCurrent ()->setTimeStepSize (0.0005);
	createBreakingDam();
}

void render ()
{
	MiniGL::coordinateSystem();
	
	// Draw simulation model
	
	const ParticleData &pd = model.getParticles();
	const unsigned int nParticles = pd.size();

	float surfaceColor[4] = { 0.2f, 0.6f, 0.8f, 1 };
	float speccolor[4] = { 1.0, 1.0, 1.0, 1.0 };
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, surfaceColor);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, surfaceColor);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, speccolor);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100.0);
	glColor3fv(surfaceColor);

	glPointSize(4.0);

	const Real supportRadius = model.getSupportRadius();
	Real vmax = 0.4*2.0*supportRadius / TimeManager::getCurrent()->getTimeStepSize();
	Real vmin = 0.0;

	if (context_major_version > 3)
	{
		for (unsigned int i = 0; i < nParticles; i++)
		{
			Real v = pd.getVelocity(i).norm();
			v = 0.5*((v - vmin) / (vmax - vmin));
			v = min(128.0*v*v, 0.5);
			float fluidColor[4] = { 0.2f, 0.2f, 0.2f, 1.0 };
			MiniGL::hsvToRgb(0.55f, 1.0f, 0.5f + (float)v, fluidColor);
			renderSphere(pd.getPosition(i), fluidColor);
		}

// 		for (unsigned int i = 0; i < model.numBoundaryParticles(); i++)
// 			renderSphere(model.getBoundaryX(i), surfaceColor);
	}
	else
	{
		glDisable(GL_LIGHTING);
		glBegin(GL_POINTS);
		for (unsigned int i = 0; i < nParticles; i++)
		{
			Real v = pd.getVelocity(i).norm();
			v = 0.5*((v - vmin) / (vmax - vmin));
			v = min(128.0*v*v, 0.5);
			float fluidColor[4] = { 0.2f, 0.2f, 0.2f, 1.0 };
			MiniGL::hsvToRgb(0.55f, 1.0f, 0.5f + (float)v, fluidColor);

			glColor3fv(fluidColor);
			glVertex3v(&pd.getPosition(i)[0]);
		}
		glEnd();

		// 	glBegin(GL_POINTS);
		// 	for (unsigned int i = 0; i < model.numBoundaryParticles(); i++)
		// 	{
		// 		glColor3fv(surfaceColor);
		// 		glVertex3fv(&model.getBoundaryX(i)[0]);
		// 	}
		// 	glEnd();

		glEnable(GL_LIGHTING);
	}



	float red[4] = { 0.8f, 0.0f, 0.0f, 1 };
	for (unsigned int j = 0; j < selectedParticles.size(); j++)
	{
		MiniGL::drawSphere(pd.getPosition(selectedParticles[j]), 0.08f, red);
	}

	MiniGL::drawTime( TimeManager::getCurrent ()->getTime ());
}

/*---------------------------------------------------------------------------------------------------*/
/*--------------------------------����SDF��ģ�ͽ������Ӳ���-------zzl-2019-1-18-start------------------*/
/*---------------------------------------------------------------------------------------------------*/

//���ݾ��볡��ģ�ͽ������Ӳ���
void generateParticleForModel()
{
	char fileName[256];
	sprintf_s(fileName, "cloud4learning\\%s.dat", "tran-scale-bunny");

	Vector3r domMax = sdf.domain().max();
	Vector3r domMin = sdf.domain().min();
	std::vector<Vector3r> particlesPos;
	unsigned int surfaceParticelNum = 0;
	double diam = 0.04;
	for(float x = domMin.x()-diam;x<domMax.x()+ diam;x=x+ diam)
		for(float y = domMin.y()- diam; y<domMax.y()+ diam;y+= diam)
			for (float z = domMin.z()- diam; z < domMax.z()+ diam; z += diam)
			{
				Vector3r pos_i(x, y, z);
				float sdf_i = sdf.interpolate(0, pos_i);
				if (sdf_i < 0)
				{
					particlesPos.push_back(pos_i);
				}
				if (sdf_i<0 && sdf_i>-diam * 3)
				{
					surfaceParticelNum++;
				}
			}
	//std::cout << particlesPos.size() << "  " << surfaceParticelNum << std::endl;
	FILE *out;
	if ((out = fopen(fileName, "wb")) == NULL)
	{
		std::cerr << "Can not open file for write : " << fileName << std::endl;
		return;
	}
	unsigned int particleNum = particlesPos.size();
	//д��������
	fwrite(&particleNum, sizeof(unsigned int), 1, out);
	//д����������
	fwrite(&surfaceParticelNum, sizeof(unsigned int), 1, out);
	for (unsigned int i = 0; i < particleNum; i++)
	{
		Vector3r posTemp = particlesPos[i];
		float x = (float)posTemp.x();
		float y = (float)posTemp.y();
		float z = (float)posTemp.z();
		//std::cout << posTemp.x() << " " << posTemp.y() << " " << posTemp.z() << std::endl;
		//std::cout << x << " " << y << " " << z << std::endl;
		fwrite(&x, sizeof(float), 1, out);
		fwrite(&y, sizeof(float), 1, out);
		fwrite(&z, sizeof(float), 1, out);
	}
	fclose(out);
}

/*---------------------------------------------------------------------------------------------------*/
/*----------------------------------------------zzl-2019-1-18-end----------------------------------*/


/*----------------------------------zzl-2019-1-21------start-------------------------------------
   -------------------------------���������ÿ������Ӻ�Ŀ������------------------------------------
-----------------------------------------------------------------------------------------------*/

//�ӳ�ʼ��״�������������������ÿ������ӣ�����֤���Ӽ��
//��Ŀ����״��������������������Ŀ������
void generateControlAndTargetParticle(vector<Vector3r> &iniShapeParticle, vector<Vector3r>&tarShapeParticle, vector<unsigned int> &controlParticleIndex, vector<unsigned int> &targetParticleIndex)
{
	//���������Ŀ���������N
	unsigned int controlParticleNum = 8000;
	unsigned int iniParitlceNum, tarParticleNum;

	unsigned int particleNum;
	//��ʼ��״����������
	particleNum = iniShapeParticle.size(); 

	iniParitlceNum = 0; 
	//���Ȳ�����������
	for (int index_Ini = 0; index_Ini < iniShapeParticle.size(); index_Ini=index_Ini+2)
	{
		controlParticleIndex.push_back(index_Ini);
		iniParitlceNum++;
	}
	

	//������ѡĿ������
	//Ŀ����״����������
	particleNum = tarShapeParticle.size(); //��ʼ��״����������
	tarParticleNum = 0;
	for (int index_Tar = 0; index_Tar < tarShapeParticle.size(); index_Tar = index_Tar + 2)
	{
		tarParticleNum++;
		targetParticleIndex.push_back(index_Tar);
	}
	int diff_ini_tar = tarParticleNum - iniParitlceNum;
	//Ŀ������<�������ӣ���Ҫ����Ŀ������
	if (diff_ini_tar < 0)
	{
		for(int add_index=0; add_index<-diff_ini_tar;add_index++)
		{
			//��������������Ա��Ϊi�������Ƿ�����Ϊ��������
			random_device gen;
			unsigned int i = gen() % (particleNum);
			//������������λ��
			Vector3r pos_i;
			pos_i = tarShapeParticle[i];

			unsigned int flag = 0; //��ǣ���¼�Ƿ�ͨ���������
			//���������λ�ý��в��ԣ���ֹ���Ѵ��ڵ�Ŀ�����ӳ�ͻ
			for (int j = 0; j < targetParticleIndex.size(); j++)
			{
				unsigned int index_j = targetParticleIndex[j];
				Vector3r pos_j = tarShapeParticle[index_j];
				double dis_i_j = (pos_j - pos_i).norm();
				//�������
				if (dis_i_j < 2 * particleRadius)
				{
					flag++;
					break;
				}
			}

			if (flag <= 0)
			{
				targetParticleIndex.push_back(i);
				tarParticleNum++;
			}
		}
	}
	//Ŀ������>�������ӣ�
	if (diff_ini_tar > 0)
	{
		for (int delete_Index = 0; delete_Index < diff_ini_tar; delete_Index++)
		{
			random_device gen;

			unsigned int k = gen() % (tarParticleNum);
			//ɾ��ָ��λ�õ�Ŀ������
			targetParticleIndex.erase(targetParticleIndex.begin() + k);
			tarParticleNum--;
		}
	}
}

void matchControlAndTargetParticles(vector<Vector3r> &iniShapeParticle, vector<Vector3r>&tarShapeParticle, vector<unsigned int> &controlParticleIndex, vector<unsigned int> &targetParticleIndex)
{
	//��������
	unsigned int optNum = 80000;
	unsigned int controlNum = controlParticleIndex.size();
	for (int iterNum = 0; iterNum < optNum; iterNum++)
	{
		
		//��������[0-controlNum-1]�������
	/*	int sand = time(NULL);
		unsigned int i = (sand + rand()) % controlNum;
		unsigned int j = (sand + rand()) % controlNum;*/

		random_device gen;
		unsigned int i = gen() % controlNum;
		unsigned int j = gen() % controlNum;
		//std::cout << i << " " << j << std::endl;
		//����i��jȷ����������
		unsigned int index_i = controlParticleIndex[i];
		unsigned int index_j = controlParticleIndex[j];

		//����������ö�Ӧ����λ��
		Vector3r pos_i = iniShapeParticle[index_i];
		Vector3r pos_j = iniShapeParticle[index_j];

		//���i��j��Ӧ��Ŀ����������
		unsigned int targetIndex_i = targetParticleIndex[i];
		unsigned int targetIndex_j = targetParticleIndex[j];

		//���Ŀ�����ӵ�λ��
		Vector3r posTarget_i = tarShapeParticle[targetIndex_i];
		Vector3r posTarget_j = tarShapeParticle[targetIndex_j];

		double oldDistance, newDistance;
		//�ֱ�����ʼ�ͽ���i��j��Ӧ��Ŀ�����Ӻ�ľ���ֵ
		oldDistance = (posTarget_i - pos_i).norm() + (posTarget_j - pos_j).norm();
		newDistance = (posTarget_i - pos_j).norm() + (posTarget_j - pos_i).norm();

		if (newDistance < oldDistance)
		{
			unsigned int temp;
			temp = targetParticleIndex[i];
			targetParticleIndex[i] = targetParticleIndex[j];
			targetParticleIndex[j] = temp;
		}

		
	}
	//float sumDis = 0.0;
	//for (int i = 0; i < controlNum; i++)
	//{
	//	Vector3r pos_con = iniShapeParticle[controlParticleIndex[i]];
	//	Vector3r pos_tar = tarShapeParticle[targetParticleIndex[i]];
	//	sumDis += (pos_con - pos_tar).norm();
	//}
	//std::cout << sumDis / controlNum << std::endl;
	//std::cout << "ok" << std::endl;
}

//Ϊʵ�ֻ��ڶ�Ӧ���ӿ��Ʒ�����״�ݻ���������������ʵ�ִ��ļ��м�����������
//flag:ΪԴ��״��Ŀ����״�ı�ǣ�0-Դ��1-Ŀ�꣩
void addParticleForControl(std::vector<Vector3r> &fluidParticles, unsigned int flag)
{
	FILE* fp = NULL;
	if (flag == 0)
	{
		fp = fopen(iniShapeName.c_str(), "rb");
		if (!fp) return;
	}
	else if(flag == 1)
	{
		fp = fopen(tarShapeName.c_str(), "rb");
		if (!fp) return;
	}
	else
	{
		std::cout << "the flag should be 0 or 1" << std::endl;
		return;
	}

	int Num;
	fread(&Num, sizeof(int), 1, fp); //��ȡ�ܵ�������
	oriParticleNum = Num;
	fread(&oriSurfaceParticleNum, sizeof(int), 1, fp); //��ȡ����������

	unsigned int needDeleteNum = 0;
	if (flag == 0)
	{
		needDeleteNum = oriParticleNum - ConstParticleNum;
		fluidParticles.resize(ConstParticleNum); //�����������������Ĵ�С
	}
	else if (flag == 1)
	{
		fluidParticles.resize(oriParticleNum);
	}
	else
	{
		std::cout << "the flag should be 0 or 1" << std::endl;
		return;
	}

	std::vector<Vector3r> deleteParticle;
	//�жϳ�ʼ�������Ƿ���ڷ����ݻ������������
	if (needDeleteNum < 0)
	{
		std::cout << "oriParticleNum is lesser than ConstParticleNum" << std::endl;
		return;
	}

	float x, y, z;
	int addParticleCnt = 0;
	int deleteParticleCnt = 0; //ɾ�����Ӽ�����
	double sdfDis;
	double disThreshold = -1.0 * 3 * 2 * particleRadius; //���������ֵ
	for (int i = 0; i < Num; i++)
	{
		//2019-1-13-���ӵ����겻һ��(�ļ��е���������-������������ӵ�����)x-z,y-x,z-y
		/*fread(&z, sizeof(float), 1, fp);
		fread(&x, sizeof(float), 1, fp);
		fread(&y, sizeof(float), 1, fp);*/

		fread(&x, sizeof(float), 1, fp);
		fread(&y, sizeof(float), 1, fp);
		fread(&z, sizeof(float), 1, fp);

		//std::cout << x << " " << y << " " << z << std::endl;
		Vector3r pos = Vector3r(x, y, z);
		sdfDis = oriSdf.interpolate(0, pos);

		double minDis = 9999;
		Vector3r minPos;
		//���������Ƿ���Ҫ���ӵ�ģ����
		if (sdfDis < disThreshold && deleteParticleCnt<needDeleteNum)
		{
			//���ӵ�һ����Ҫɾ��������
			if (deleteParticleCnt == 0)
			{
				deleteParticle.push_back(pos);
				deleteParticleCnt++;
				continue;
			}
			else //����
			{
				for (int j = 0; j < deleteParticle.size(); j++)
				{
					Vector3r delePos = deleteParticle[j];
					double dis = (delePos - pos).norm();
					if (dis < minDis)
					{
						//��¼����ɾ�����ӵ���С�����λ��
						minDis = dis;
						minPos = delePos;
					}
				} //end for(j)
				  //��ֹĳЩ����ɾ�����ӹ�������о�������
				if (minDis < 1000 && minDis > 4.0*particleRadius)
				{
					deleteParticle.push_back(minPos);
					deleteParticleCnt++;
					continue;
				}
			}

		}
		fluidParticles[addParticleCnt] = pos;
		addParticleCnt++;
	}
	
	LOG_INFO << "Number of particles: " << Num;
	
	fclose(fp);

}
/*----------------------------------zzl-2019-1-21------end-------------------------------------*/

//////////////////---zzl---------start----2018-7-13--------------
/** ��.dat�ļ��ж�ȡ�Ƶ����ݣ����������ӣ�ֻ��������λ�����ԣ�
* ��ȡĿ��ģ�����ɵ��������ݣ����Ŀ��ģ����Ҫ�����������ͱ���������
* ������fluidParticles(��ʼ��״�����ӣ� targetParticles(Ŀ����״�е�Ŀ������,���ڿ���)
*/
void addParticleFromFile(std::vector<Vector3r> &fluidParticles, std::vector<Vector3r> &targetParticles)
{
	FILE* fp = NULL;
	/*char strname[256];
	sprintf(strname, "clouds\\cloud64.dat");
	fp = fopen(strname, "rb");*/
	fp = fopen(iniShapeName.c_str(), "rb");
	if (!fp) return;

	int Num;
	fread(&Num, sizeof(int), 1, fp); //��ȡ�ܵ�������
	oriParticleNum = Num;
	fread(&oriSurfaceParticleNum, sizeof(int), 1, fp); //��ȡ����������

	unsigned int needDeleteNum = oriParticleNum - ConstParticleNum;
	std::vector<Vector3r> deleteParticle;
	//�жϳ�ʼ�������Ƿ���ڷ����ݻ������������
	if (needDeleteNum < 0)
	{
		std::cout << "oriParticleNum is lesser than ConstParticleNum" << std::endl;
		return;
	}
	
	fluidParticles.resize(ConstParticleNum); //�����������������Ĵ�С

	std::cout << Num << std::endl;
	
	float x, y, z;
	int addParticleCnt = 0;
	int deleteParticleCnt = 0; //ɾ�����Ӽ�����
	double sdfDis;
	double disThreshold = -1.0 * 3 * 2 * particleRadius; //���������ֵ
	for (int i = 0; i < Num; i++)
	{
		//2019-1-13-���ӵ����겻һ��(�ļ��е���������-������������ӵ�����)x-z,y-x,z-y
		/*fread(&z, sizeof(float), 1, fp);
		fread(&x, sizeof(float), 1, fp);
		fread(&y, sizeof(float), 1, fp);*/

		fread(&x, sizeof(float), 1, fp);
		fread(&y, sizeof(float), 1, fp);
		fread(&z, sizeof(float), 1, fp);

		//std::cout << x << " " << y << " " << z << std::endl;
		Vector3r pos = Vector3r(x, y, z);
		sdfDis = oriSdf.interpolate(0, pos);
		
		double minDis = 9999;
		Vector3r minPos;
		//���������Ƿ���Ҫ���ӵ�ģ����
		if (sdfDis < disThreshold && deleteParticleCnt<needDeleteNum)
		{
			//���ӵ�һ����Ҫɾ��������
			if (deleteParticleCnt == 0) 
			{
				deleteParticle.push_back(pos);
				deleteParticleCnt++;
				continue;
			}
			else //����
			{
				for (int j = 0; j < deleteParticle.size(); j++)
				{
					Vector3r delePos = deleteParticle[j];
					double dis = (delePos - pos).norm();
					if (dis < minDis)
					{
						//��¼����ɾ�����ӵ���С�����λ��
						minDis = dis;
						minPos = delePos;
					}
				} //end for(j)
				  //��ֹĳЩ����ɾ�����ӹ�������о�������
				if (minDis < 1000 && minDis > 4.0*particleRadius)
				{
					deleteParticle.push_back(minPos);
					deleteParticleCnt++;
					continue;
				}
			}
			
		}
		fluidParticles[addParticleCnt] = pos;
		addParticleCnt++;
	}
	std::cout << fluidParticles.size() << std::endl;
	//ͨ��ɾ�����ӣ�ʹ�������� = ConstParticleNum������ָ����������������
	
	LOG_INFO << "Number of particles: " << Num;
	//std::cout << fluidParticles.size() << std::endl;

	//std::cout << oriSurfaceParticleNum << std::endl;
	//std::cout << deleteParticleCnt << std::endl;
	//std::cout << "============================" << std::endl;
	fclose(fp);

	//��ȡĿ����״��Ҫ�������������ͱ���������
	FILE* targetFp = NULL;
	targetFp = fopen(tarShapeName.c_str(), "rb");
	if (!fp) return;
	fread(&tarParticleNum, sizeof(int), 1, fp); //��ȡ�ܵ�������
	fread(&tarSurfaceParticleNum, sizeof(int), 1, fp); //��ȡ����������

	for (int j = 0; j < tarParticleNum; j=j+2)
	{
		fread(&x, sizeof(float), 1, fp);
		fread(&y, sizeof(float), 1, fp);
		fread(&z, sizeof(float), 1, fp);
		Vector3r pos = Vector3r(x, y, z);
		targetParticles.push_back(pos);
	}
	//std::cout << tarSurfaceParticleNum << std::endl;
	//std::cout << tarParticleNum << std::endl;
	//std::cout << "============================" << std::endl;
	fclose(targetFp);
}
//////////////////---zzl---------end----2018-7-13--------------


/** Create a breaking dam scenario
*/
void createBreakingDam()
{
	LOG_INFO << "Initialize fluid particles";
	const Real diam = 2.0*particleRadius;

	//const Real startX = -0.5*containerWidth + diam;
	//const Real startY = diam;
	//const Real startZ = -0.5*containerDepth + diam;

	const Real startX = -0.5;
	const Real startY = -0.3;
	const Real startZ = -0.5;
	const Real yshift = sqrt(3.0) * particleRadius;

	//std::cout << "x:"<<startX <<"   y:"<< startY <<"  z:"<< startZ << std::endl;

	std::vector<Vector3r> fluidParticles; //���������ӵ�������ֻ��������ӵ�λ��
	
	std::vector<Vector3r> targetParticles;//���Ŀ����״�е�Ŀ������

	/*-------�������ӷ�---------zzl----2019-1-21----------start-------------*/
	//�������ӷ�����ԭʼ��״�����ɿ������ӣ���Ŀ����״������Ŀ�����ӣ����ҳ��������Ӻ�Ŀ�����ӵĶ�Ӧ��ϵ���п���
	std::vector<Vector3r> targetShapeParticles;//�ñ������Ŀ����״����������
	//��������ͬ�Ŀ������Ӻ�Ŀ������Ϊ��Ӧ��ϵ
	std::vector<unsigned int> controlParticleIndex; //��ſ������ӵ�����
	std::vector<unsigned int> targetParticleIndex;  //���Ŀ�����ӵ�����
    /*-------�������ӷ�---------zzl----2019-1-21----------end--------------*/

	////---------------zl--------------------//
	////---------�������ӷ�ʽ 1---------------//
	////---�������ھ��ȷ�������-------------//
	////------------------18-5-5-21:00------//
	/*fluidParticles.resize(width*height*depth);
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)width; i++)
		{
			for (unsigned int j = 0; j < height; j++)
			{
				for (unsigned int k = 0; k < depth; k++)
				{
					fluidParticles[i*height*depth + j*depth + k] = diam*Vector3r((Real)i, (Real)j, (Real)k) + Vector3r(startX, startY, startZ);
				}
			}
		}
	}*/


	/*fluidParticles.resize(width*height * depth);
	#pragma omp parallel default(shared)
	{
	#pragma omp for schedule(static)  
		for (int i = 0; i < (int)width; i++)
		{
			for (unsigned int j = 0; j < height; j++)
			{
				for (unsigned int k = 0; k < depth; k++)
				{
					fluidParticles[i*(height)*depth + j*depth + k] = diam*Vector3r((Real)i, (Real)j, (Real)k) + Vector3r(startX, startY, startZ);
				}
			}
		}
	}*/

	///////////------zzl-----start-----2018-7-14----------
	////---------�������ӷ�ʽ 2---------------//
	//Add�� load particles from .dat file.

	addParticleFromFile(fluidParticles,targetParticles);

	//����Ŀ�����Ӻ���Ҫִ�������������--zzl
	//model.setConstParCnt(ConstParticleNum); //����ģ����Ҫ������������
	//model.setIniShpaePar(oriParticleNum, oriSurfaceParticleNum); //���ó�ʼ��״�������������������ͱ���������
	//model.setTarShpaePar(tarParticleNum, tarSurfaceParticleNum); //����Ŀ����״�������������������ͱ���������
	///////////------zzl-----end-----2018-7-14----------

	/*----------------zzl-----------2019-1-16-------start------*/
	//����Ŀ������
	//model.resizeTargetParticles(targetParticles.size());
	//for (unsigned int k = 0; k < targetParticles.size(); k++)
	//{
	//	Vector3r temPos = targetParticles[k];
	//  model.setTargetParticles(k, temPos);
	//}
	/*----------------zzl-----------2019-1-16-------end------*/


	/*---------------�������ӷ�--------zzl---2019-1-21-----start------------*/
	//addParticleForControl(fluidParticles, 0); //
	addParticleForControl(targetShapeParticles, 1);
	generateControlAndTargetParticle(fluidParticles, targetShapeParticles, controlParticleIndex, targetParticleIndex);
	matchControlAndTargetParticles(fluidParticles, targetShapeParticles, controlParticleIndex, targetParticleIndex);
	//��Ŀ�����ӿ�����model
	model.resizeTargetParticles(targetShapeParticles.size());
	for (unsigned int k = 0; k < targetShapeParticles.size(); k++)
	{
		Vector3r temPos = targetShapeParticles[k];
	  model.setTargetParticles(k, temPos);
	}
	//���������Ӻ�Ŀ����������������model
	unsigned int size = controlParticleIndex.size();
	model.setSizeOfControlParticleIndex(size);
	model.setSizeOfTargetParticleIndex(size);
	for (int i = 0; i < size; i++)
	{
		unsigned int con_index = controlParticleIndex[i];
		unsigned int tar_index = targetParticleIndex[i];
		model.setControlParticleIndex(i, con_index);
		model.setTargetParticleIndex(i, tar_index);
	}
	/*---------------�������ӷ�--------zzl---2019-1-21-----end-------------*/

	
	model.setParticleRadius(particleRadius);

	//----------------set environmentGrid------------zzl------8-20-----
	model.getEnvironmentGrid().initGrid(xHalfGridNum,yHalfGridNum,zHalfGridNum, 2 * particleRadius, 2 * particleRadius, 2 * particleRadius);
	//-----------------------------------------------end------8-20-----

	std::vector<Vector3r> boundaryParticles;
	initBoundaryData(boundaryParticles);

	model.initModel((unsigned int)fluidParticles.size(), fluidParticles.data(), (unsigned int)boundaryParticles.size(), boundaryParticles.data());

	//LOG_INFO << "Number of particles: " << width*height*depth;

	//���ļ�������SDF
	//std::string name = "clouds\\box.cdf";
	//Discregrid::CubicLagrangeDiscreteGrid sdf = simulation.computeSDF(name);
	
	//���ݷ��ž��볡�����ݶȳ�
	//sdfGraField = PBD::SDFGradientField(sdf);
	//sdfGraField.computPotentialField(sdf); //�������ɵ��ݶȳ������Ƴ�

	//����ÿ����������λ�õľ���
}


void addWall(const Vector3r &minX, const Vector3r &maxX, std::vector<Vector3r> &boundaryParticles)
{
	const Real particleDistance = 2.0*model.getParticleRadius();

	const Vector3r diff = maxX - minX;
	const unsigned int stepsX = (unsigned int)(diff[0] / particleDistance) + 1u; //+1u��Ϊ�˱�����Ϊ0
	const unsigned int stepsY = (unsigned int)(diff[1] / particleDistance) + 1u;
	const unsigned int stepsZ = (unsigned int)(diff[2] / particleDistance) + 1u;

	const unsigned int startIndex = (unsigned int) boundaryParticles.size();
	boundaryParticles.resize(startIndex + stepsX*stepsY*stepsZ);

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)  
		for (int j = 0; j < (int)stepsX; j++)
		{
			for (unsigned int k = 0; k < stepsY; k++)
			{
				for (unsigned int l = 0; l < stepsZ; l++)
				{
					const Vector3r currPos = minX + Vector3r(j*particleDistance, k*particleDistance, l*particleDistance);
					boundaryParticles[startIndex + j*stepsY*stepsZ + k*stepsZ + l] = currPos;
				}
			}
		}
	}
}

void initBoundaryData(std::vector<Vector3r> &boundaryParticles)
{
	const Real x1 = -containerWidth / 2.0;
	const Real x2 = containerWidth / 2.0;
	const Real y1 = 0.0;
	const Real y2 = containerHeight;
	const Real z1 = -containerDepth / 2.0;
	const Real z2 = containerDepth / 2.0;

	const Real diam = 2.0*particleRadius;

	// Floor
	addWall(Vector3r(x1, y1, z1), Vector3r(x2, y1, z2), boundaryParticles);
	// Top
	addWall(Vector3r(x1, y2, z1), Vector3r(x2, y2, z2), boundaryParticles);
	// Left
	addWall(Vector3r(x1, y1, z1), Vector3r(x1, y2, z2), boundaryParticles);
	// Right
	addWall(Vector3r(x2, y1, z1), Vector3r(x2, y2, z2), boundaryParticles);
	// Back
	addWall(Vector3r(x1, y1, z1), Vector3r(x2, y2, z1), boundaryParticles);
	// Front
	addWall(Vector3r(x1, y1, z2), Vector3r(x2, y2, z2), boundaryParticles);
}


void createSphereBuffers(Real radius, int resolution)
{
	Real PI = static_cast<Real>(M_PI);
	// vectors to hold our data
	// vertice positions
	std::vector<Vector3r> v;
	// normals
	std::vector<Vector3r> n;
	std::vector<unsigned short> indices;

	// initiate the variable we are going to use
	Real X1, Y1, X2, Y2, Z1, Z2;
	Real inc1, inc2, inc3, inc4, radius1, radius2;

	for (int w = 0; w < resolution; w++)
	{
		for (int h = (-resolution / 2); h < (resolution / 2); h++)
		{
			inc1 = (w / (Real)resolution) * 2 * PI;
			inc2 = ((w + 1) / (Real)resolution) * 2 * PI;
			inc3 = (h / (Real)resolution)*PI;
			inc4 = ((h + 1) / (Real)resolution)*PI;

			X1 = sin(inc1);
			Y1 = cos(inc1);
			X2 = sin(inc2);
			Y2 = cos(inc2);

			// store the upper and lower radius, remember everything is going to be drawn as triangles
			radius1 = radius*cos(inc3);
			radius2 = radius*cos(inc4);

			Z1 = radius*sin(inc3);
			Z2 = radius*sin(inc4);

			// insert the triangle coordinates
			v.push_back(Vector3r(radius1*X1, Z1, radius1*Y1));
			v.push_back(Vector3r(radius1*X2, Z1, radius1*Y2));
			v.push_back(Vector3r(radius2*X2, Z2, radius2*Y2));

			indices.push_back((unsigned short)v.size() - 3);
			indices.push_back((unsigned short)v.size() - 2);
			indices.push_back((unsigned short)v.size() - 1);

			v.push_back(Vector3r(radius1*X1, Z1, radius1*Y1));
			v.push_back(Vector3r(radius2*X2, Z2, radius2*Y2));
			v.push_back(Vector3r(radius2*X1, Z2, radius2*Y1));

			indices.push_back((unsigned short)v.size() - 3);
			indices.push_back((unsigned short)v.size() - 2);
			indices.push_back((unsigned short)v.size() - 1);

			// insert the normal data
			n.push_back(Vector3r(X1, Z1, Y1));
			n.push_back(Vector3r(X2, Z1, Y2));
			n.push_back(Vector3r(X2, Z2, Y2));
			n.push_back(Vector3r(X1, Z1, Y1));
			n.push_back(Vector3r(X2, Z2, Y2));
			n.push_back(Vector3r(X1, Z2, Y1));
		}
	}

	for (unsigned int i = 0; i < n.size(); i++)
		n[i].normalize();


	glGenBuffersARB(1, &vertexbuffer);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vertexbuffer);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, v.size() * sizeof(Vector3r), &v[0], GL_STATIC_DRAW);

	glGenBuffersARB(1, &normalbuffer);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalbuffer);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, n.size() * sizeof(Vector3r), &n[0], GL_STATIC_DRAW);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	// Generate a buffer for the indices as well
	glGenBuffersARB(1, &elementbuffer);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, elementbuffer);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, indices.size() * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);

	// store the number of indices for later use
	vertexBufferSize = (unsigned int)indices.size();

	// clean up after us
	indices.clear();
	n.clear();
	v.clear();
}

void renderSphere(const Vector3r &x, const float color[])
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);


	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vertexbuffer);
	glVertexPointer(3, GL_REAL, 0, 0);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalbuffer);
	glNormalPointer(GL_REAL, 0, 0);

	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, elementbuffer);

	glPushMatrix();
	glTranslated(x[0], x[1], x[2]);
	glDrawElements(GL_TRIANGLES, (GLsizei)vertexBufferSize, GL_UNSIGNED_SHORT, 0);
	glPopMatrix();
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void releaseSphereBuffers()
{
	if (elementbuffer != 0)
	{
		glDeleteBuffersARB(1, &elementbuffer);
		elementbuffer = 0;
	}
	if (normalbuffer != 0)
	{
		glDeleteBuffersARB(1, &normalbuffer);
		normalbuffer = 0;
	}
	if (vertexbuffer != 0)
	{
		glDeleteBuffersARB(1, &vertexbuffer);
		vertexbuffer = 0;
	}
}


void TW_CALL setTimeStep(const void *value, void *clientData)
{
	const Real val = *(const Real *)(value);
	TimeManager::getCurrent()->setTimeStepSize(val);
}

void TW_CALL getTimeStep(void *value, void *clientData)
{
	*(Real *)(value) = TimeManager::getCurrent()->getTimeStepSize();
}

void TW_CALL setVelocityUpdateMethod(const void *value, void *clientData)
{
	const short val = *(const short *)(value);
	((TimeStepFluidModel*)clientData)->setVelocityUpdateMethod((unsigned int)val);
}

void TW_CALL getVelocityUpdateMethod(void *value, void *clientData)
{
	*(short *)(value) = (short)((TimeStepFluidModel*)clientData)->getVelocityUpdateMethod();
}

void TW_CALL setViscosity(const void *value, void *clientData)
{
	const Real val = *(const Real *)(value);
	((FluidModel*)clientData)->setViscosity(val);
}

void TW_CALL getViscosity(void *value, void *clientData)
{
	*(Real *)(value) = ((FluidModel*)clientData)->getViscosity();
}

