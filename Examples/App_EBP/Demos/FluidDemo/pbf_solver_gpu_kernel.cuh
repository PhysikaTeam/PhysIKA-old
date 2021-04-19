#ifndef _pbf_solver_kernel_h_
#define _pbf_solver_kernel_h_

#include "vector_types.h"


typedef unsigned int uint;

struct SimParams
{
	//informtion about grid and cell
	float cellSize; //��Ԫ����,x,y,z������ͬ=2*particleRadius
	int3 gridSize; //x,y,z�����ϰ������ٸ�����,һ��|x=y=z|
	int numGridCells; //��������
	float3 gridOrigin;//����ռ��x\y\z����С����ֵ

	//information about particle
	float particleRadius;
	float3 acceleation; //���ٶ�
	int maxNumOfNeighbors; //���������

	double kernelRadius; //�˺�����֧�ְ뾶
	double density0; //rest density
	double gasConstantK; //���峣����������ѹ��
};

#endif // !_pbf_solver_kernel_h_
