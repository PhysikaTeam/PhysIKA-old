#pragma once
#include"node.h"
#include"contactpair.h"
#include "testpair.h"
#include "unified_contact.h"

struct NodeCuda;
struct ContactNodeManager;
struct ContactPairCuda;
struct TestPairCuda;
struct SubDomain;

typedef struct ContactNodeCuda
{
	NodeCuda *node;
	int nodeId;

	int gpu_id;

	/**
	itct表示该点正处于碰撞中
	*/
	int itct;

	/**
	istp表示由接触对退化成测试对
	int istp;*/

	int id;

	/**
	属于该接触点的测试对
	*/
	TestPairCuda test_pair;

	/**
	包含该节点的接触块
	*/
#define max_parent_num 12
	SegmentCuda* parent_segment[max_parent_num];
	int parent_segment_num;

	/**
	属于该接触点的碰撞对
	*/
	ContactPairCuda contact_pair;

	/**
	接触节点所归属的接触面
	*/
	int surfaceId;

	/**
	罚因子
	*/
	double interfaceStiff;

	/**
	应用子域法时该碰撞点所属的子域,为Gpu上的指针
	*/
	SubDomain *parent_sub_domain;

	/**
	gpu接触点复制
	*/
	__host__  __device__ void gpuContactNodeCudaCpy(NodeCuda *node_array);

	/**
	计算本接触点的速度矢量的模
	*/
	__host__ __device__ double calSpeedMode();

	/**
	整合接触力
	*/
	__host__ __device__ void singleInterContactForce();

	/**
	清除上一步骤的接触力
	*/
	__host__ __device__ void clearOldContactForce();

	/**
	重置所有标签
	*/
	__host__ __device__ void clearFlag();

	/**
	接触点罚因子求解
	*/
	//存在一定错误
	/*__host__ __device__ void calPenaltyFactor(const double dt, const double scale_factor);*/

} ContactNodeCuda;