#pragma once
#include"testpair.h"
#include"contactpair.h"
#include"unified_contact.h"
#include"contactnodemanager.h"
#include"surface.h"
#include "contactmanager.h"

#define DefenNodeDecople

struct UnifiedContactManager;
struct ContactPairCuda;
struct TestPairCuda;

typedef struct PairManager
{
	typedef enum SearchType
	{
		OneDirec, TwoDirec
	}SearchType;

	typedef enum ContactForceMethod
	{
		DefenceNode, Penalty
	}ContactForceMethod;

	/**
	标记接触对搜寻及接触力计算是否是双向的
	*/
	SearchType search_type;

	/**
	标价接触力的计算方法
	*/
	ContactForceMethod cal_con_method;

	/**
	gpu测试对
	*/
	TestPairCuda** test_pair_array_gpu;

	/*
	* 多gpu测试对
	*/
	vector<TestPairCuda**> testPairArrayMultiGpu;
	vector<int> testPairNumMultiGpu;

	/**
	cpu测试对
	*/
	vector<TestPairCuda*> test_pair_array_cpu;

	/**
	总的接触点数目
	*/
	int contact_node_num;

	/**
	上一步的接触对数目，为下一步的接触后搜索准备
	*/
	int previous_contact_pair_num;

	/**
	gpu端测试对数目计数
	*/
	int* test_pair_num_gpu;
	int* test_pair_num_cpu;

	/*
	* 多gpu进入下一计算的测试对数目
	*/
	vector<int> testPairForNextCalNumCpu;
	vector<int*> testPairForNextCalNumMultiGpu;

	/**
	gpu端碰撞对
	*/
	ContactPairCuda** contact_pair_array_gpu;

	/*
	* 多Gpu接触对
	*/
	vector<ContactPairCuda**> contactPairArrayMultiGpu;
	vector<int> contactPairNumMultiGpu;

	/**
	cpu端碰撞对
	*/
	vector<ContactPairCuda*> contact_pair_array_cpu;

	/**
	gpu端碰撞对数目计数
	*/
	int* contact_pair_num_gpu;
	int* contact_pair_num_cpu;

	/*
	* 多gpu参与下一步计算的接触对数目
	*/
	vector<int> contactPairNumForNextCalCpu;
	vector<int*> contactPairNumForNextCalMultiGpu;

	int minPredictCycles;

	/**
	测试对排序
	*/
	void testPairSortGpu();
	void testPairSortCpu();

	/*
	* 多gpu测试对排序
	*/
	void testPairSortMultiGpu(int gpu_id);

	/**
	碰撞对排序
	*/
	void contactPairSortGpu();
	void contactPairSortCpu();

	/*
	* 多gpu碰撞排序
	*/
	void contactPairSortMultiGpu(int gpu_id);

	/**
	接触对接触力计算
	*/
	void defenCalContactForGpu(double dt, int cyc_num, ContactNodeManager *contact_node_manager);
	void contactForCalCpu(double dt, int cyc_num);

	/*
	* 多gpu计算接触力
	*/
	void defenCalContactMultiGpu(vector<int> gpuIdArray, vector<cudaStream_t> streamArray,
		double dt, int cyc_num, ContactNodeManager *cnNdMag);

	/**
	生成接触对
	*/
	void produceContactPairGpu(int cyc_num, int mstp);
	void produceContactPAirCpu(int cyc_num, int mstp);

	/*
	* 多gpu生成接触对
	*/
	void produceContactPairMultiGpu(vector<int> gpuIdArray, vector<cudaStream_t> streamArray, int cyc_num, int mstp);

	/**
	接触后搜索
	*/
	void searchAfterContactGpu();
	void searchAfterContactCpu();
	void searchAfterContactMultiGpu(vector<int> gpuIdArray, vector<cudaStream_t> streamArray);

	/*
	 * 顺序式接触后搜寻
	 */
	void searchAfterContactGpuBySequential();

	/**
	接触前搜索
	*/
	void searchBeforeContactGpu(UnifiedContactManager *unifiedContactManager);

	void searchBeforeContactGpuImprove(UnifiedContactManager *unifiedContactManager);

	/*
	* 多gpu一体化法接触算法
	*/
	void searchBeforeContactMultiGpu(UnifiedContactManager* unifiedContactManager, vector<int> gpuIdArray, vector<cudaStream_t> steamArray);

	/**
	双向接触前搜索
	*/
	void searchBeforeContactBilateralGpu(UnifiedContactManager *unifiedContactManager);

	/**
	由已有的接触对生成反向的接触对
	*/
	void produceReverseContactPair();

	/**
	接触对集合生成
	*/
	void gpuContactPairSetCreate(ContactNodeManager *contact_node_manager);
	void multiGpuContactPairSetCreate(ContactNodeManager *cnNdMag, vector<int> gpuIdArray);

	/**
	gpu端测试对集合生成
	*/
	void gpuTestPairSetCreate(ContactNodeManager *contact_node_manager);
	void multiGpuTestPairSetCreate(ContactNodeManager *cnNdMag, vector<int> gpuIdArray);

	/**
	对所有的测试对进行检测，剔除掉不合理的测试对
	*/
	void gpuTestAllPair();

	/**
	清除计数变量
	*/
	void clearCount();

	/**
	gpu端统计需要进行下一步计算的节点
	*/
	void countNextTestPairCal();

	/*
	* 多gpu统计需要进行下一步计算的节点
	*/
	void countNextTestPairCalMultiGpu(vector<int> gpuIdArray, vector<cudaStream_t> streamArray);

	/**
	罚函数法求解接触力
	*/
	void penaltyCalContForGpu(double dt, int cyc_num, int penalty_method);

	/**
	分配零复制内存
	*/
	void allocZeroMemPair();
	void allocMemForMultiGpu(vector<int> gpuIdArray);

	~PairManager();
} PairManager;