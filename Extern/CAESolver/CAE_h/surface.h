#pragma once
#include"node.h"
#include"nodemanager.h"
#include"segment.h"
#include"contactnode.h"
#include"contactnodemanager.h"
#include"shell1st.h"
#include<string>

using std::string;

struct ShellElementCuda;
struct ContactNodeManager;
struct SegmentCuda;
struct ContactNodeCuda;
struct SubDomain;
struct Part;
struct FEMDynamic;

typedef struct Surface
{
	string name;

	string partName;

	vector<Part*> partArray;

	vector<int> partStoreIdArray;

	//ls:2020-04-06
	/**
	First\Second\Third\Fourth segment attribute default value
	*/
	double DA[4];
	string solver;
	//

	/*
	 *面上所有的块
	 */
	vector<SegmentCuda> segment_array;
	SegmentCuda *segment_array_gpu;

	/*
	* 多Gpu接触块
	*/
	vector<SegmentCuda*> segmentArrayMulyiGpu;

	/*
	 *面上所有的点
	 */
	vector<NodeCuda *> node_array;

	/*
	 *当面参与碰撞时，生成的碰撞点，不参与碰撞则不生成
	 */
	vector<ContactNodeCuda *> contact_node_array_cpu;
	ContactNodeCuda **contact_node_array_gpu;

	/*
	* 多gpu上的接触点
	*/
	vector<ContactNodeCuda**> contactNodeArrayMultiGpu;
	vector<int> contactNodeNumMultiGpu;

	/*
	 * 依附此面存在的子域
	 */
	int subDomainNum_;
	SubDomain* subDomainArrayCpu_;
	SubDomain* subDomainArrayGpu_;
	vector<SubDomain*> subDomainArrayMultiGpu_;

	/*
	 * 依附于面上的子域的相关数据
	 */
	int3 axi_num;//只在接触准备阶段计算一次
	double3 max_val;
	double3 min_val;
	double3 scal;

	double3 another_max_cpu;
	double3 another_min_cpu;

	double3 *max_gpu;
	double3 *min_gpu;

	double3 *max_cpu;
	double3 *min_cpu;

	double velMaxModulus;

	/*
	 * 标记该面是否求过最大速度
	 */
	bool isCalMaxVel;

	/*
	 * 标记该面是否检索过节点的尺寸
	 */
	bool isSearchNodeSize_;

	/*
	 * 标记该面是否填充过子域
	 */
	bool isFullSubDomain_;

	/*
	 * 标记该面是否检索过大小
	 */
	bool isSearchSize_;

	/*
	 * 标记是否更新过扩展域
	 */
	bool isUdateExtDomain_;

	/*
	 * 标记是否清除过一体化子域中的数据
	 */
	bool isClearSubDomainData_;

	double maxThick1;
	double *max_disp_incre;

	int id;
	int isym;
	int segment_num;

	/**
	接触面最大最小坐标
	*/
	double xmax[3];
	double xmin[3];

	/**
	接触面最大速度
	*/
	double vmax;

	/*
	 * 接触面最小边长
	 */
	double min_length;

	/**
	接触面平均边长
	*/
	double ave_length;

	/**
	接触面平均扩展域大小
	*/
	double ave_expand;

	/**
	接触面最大位移增量
	*/
	double maxDispIncre;

	Surface();

	void clearDataInSubDomainGpu();

	void clearDataInSubDomainMultiGpu(int gpuId, cudaStream_t& stream);
	/**
	分配最大位移增量的gpu内存
	*/
	void allocDispMem();

	void allocNeddMem();

	/**
	复制最大位移增量
	*/
	void cpyDispIncre(double* disp_incre);

	/**
	接触面接触块统计
	*/
	void segmentCount();

	/**
	统计本接触面有多少节点以及单个接触块有多少个节点
	*/
	void nodeCount(NodeManager &node_manager);

	/**
	求解本面所有块的法向量
	*/
	void solveAllSegmentNorVec();

	/**
	统计本接触面有多少接触点
	*/
	void contactNodeLink(ContactNodeManager *contactNodeManager);

	/**
	gpu端碰撞点生成
	*/
	void gpuContactNodeGet(ContactNodeManager* contactNodeManager);

	/*
	* 多gpu碰撞点生成
	*/
	void multiGpuContactNodeGet(vector<int> gpuIdArray, ContactNodeManager* cnNdMag);

	/**
	寻找相邻的接触块,计算平均边长
	*/
	void neighborSegmentCudaSearch();

	/*
	 * 计算所有块的扩展域尺寸
	 */
	void calAllSegmentExtTerritorySize();

	/**
	cpu端接触块生成
	*/
	void cpuSegmentCudaCreate(NodeManager *node_manager);

	/**
	gpu接触块生成
	*/
	void gpuSegmentCudaCreate(NodeManager *node_manager);

	/*
	* 多gpu接触块生成
	*/
	void multiGpuSegmentCreate(vector<int> gpuIdArray, NodeManager* ndMag = nullptr);

	/**
	壳单元生成接触面
	*/
	void shellElementToSegmentCuda(const vector<ShellElementCuda> &elementArray);

	/**
	单元集合生成接触面（壳单元）ls:2020-07-10 added
	*/
	void shellElementSetTosegmentCuda(vector<ShellElementCuda*> element_array_cpu);

	/**
	实体单元表面生成接触面
	*/
	void solidElementToSegmentCuda(Part* part_temp, NodeManager *nodeManager);

	/**
	生成接触块的扩展域
	*/
	void produceExtendedDomain();

	/**
	gpu生成接触块的扩展域
	*/
	void gpuUpdateExtendedDomainImprove();
	void gpuUpdateExtendedDomain();

	/*
	* 多gpu更新接触块的扩展域
	*/
	void multiGpuUpdateExtendedDomainImprove(vector<int> gpuIdAraray, vector<cudaStream_t> streamArray);
	void multiGpuUpdateExtendedDomain(vector<int> gpuIdAraray, vector<cudaStream_t> streamArray);
	/**
	设置接触面中所有的接触块的厚度
	*/
	void setAllSegmentCudaThick();

	/**
	指定本面中的接触块的id
	*/
	void assignSegmentId(const int global_id);

	/**
	求解接触面扩展域的平均大小
	*/
	void calAveExpandSize();

	void produceSegmentSetFromPart(FEMDynamic *domain);

	virtual ~Surface();
} Surface;