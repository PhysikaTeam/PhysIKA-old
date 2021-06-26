#pragma once
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#define TestPairOfNode 64
#define AssemblyClearFlag

struct ContactNodeCuda;
struct SegmentCuda;

/**
gpu测试对判断生成的值
*/
typedef struct TestJudgeReturn
{
	/**
	type=0：不形成接触对 =1：点对块 =2：点对边 =3：点对点
	*/
	int type;
	int edge_or_node;
	int position;
	SegmentCuda *segment;
	double dist;

}TestJudgeReturn;

typedef struct TestPairCuda
{
	/*int contact_node_id;*////@是否彻底删除还需要进一步观察

	/**
	组成测试对的相关属性
	*/
	int predict_cycle[TestPairOfNode];
	int position[TestPairOfNode];
	int relative_position[TestPairOfNode];
	SegmentCuda* test_segment[TestPairOfNode];

	/*int test_segment_id[TestPairOfNode];*////@是否彻底删除还需要进一步观察
	/*double cos_value[TestPairOfNode];*////@是否彻底删除还需要进一步观察

	/**
	有效的测试对，值不能超过TestPairOfNode
	*/
	int effective_num;

	/**
	组成测试对的接触点与接触块
	*/
	ContactNodeCuda* contactNode;

	/**
	测试对下一步计算标志
	*/
	int further_cal_sign;

	__host__ __device__ TestPairCuda();

	/**
	生成碰撞对
	*/
	__device__ void produceContactPairGpu(SegmentCuda *segment_temp, const int cycleNum, const int current_segment_id,
		TestJudgeReturn &judge_result, const int mstp);
	__host__ void produceContactPairCpu(SegmentCuda *segment_temp, const int cycleNum, const int current_segment_id,
		TestJudgeReturn &judge_result, const int mstp);

	/**
	测试对生成点对边或点对点判断
	*/
	__device__ void nodeToEdgeJudgeGpu(SegmentCuda *segment_temp, const int current_segment_id, TestJudgeReturn &judge_result,
		const int numberOfEdge, const int mstp, const double normVec[3]);
	__host__ void nodeToEdgeJudgeCpu(SegmentCuda *segment_temp, const int current_segment_id, TestJudgeReturn &judge_result,
		const int numberOfEdge, const int mstp, const double normVec[3]);

	/**
	不考虑边缘接触对接触判断的影响
	*/
	__device__ void nodeToNoThickEdgeJudgeGpu(SegmentCuda *segment_temp, const int current_segment_id, TestJudgeReturn &judge_result,
		const int numberOfEdge, const int mstp, const double normVec[3]);

	/**
	测试对生成点对块判断
	*/
	__device__ void nodeToSegmentCudaJudgeGpu(SegmentCuda *segment_temp, const int current_segment_id, TestJudgeReturn &judge_result,
		const int mstp, const double normalVec[3]);
	__host__ void nodeToSegmentCudaJudgeCpu(SegmentCuda *segment_temp, const int current_segment_id, TestJudgeReturn &judge_result,
		const int mstp, const double normalVec[3]);

	/**
	测试对进一步计算检测
	*/
	__host__ __device__ bool furtherCalculateTest();

	/**
	现有测试对对自身数目进行检测,测试通过返回true，否则false
	*/
	__host__ __device__ void testExpand();

	/**
	测试碰撞点是否穿越了接触块所在平面
	*/
	__host__ __device__ void updatePositionAfterContactBreak(int cycNum);

	/**
	测试碰撞点是否穿越了接触块所在平面
	*/
	__host__ __device__ void updatePositionAfterContactBreak(int tid, int cycnum);
}TestPairCuda;