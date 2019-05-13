        #ifndef DEF_KERN_CUDA_SWE
	#define DEF_KERN_CUDA_SWE

	#include <stdio.h>
	#include <vector_types.h>
	#include "Physika_Surface_Fuild/Surface_Fuild_Model/Simulator2.h"
	#include "Physika_Surface_Fuild/SPHsrc/fluid_defs.h"
#ifdef HYBRID
	#include "Physika_Surface_Fuild/SPHsrc/fluid_system_host.cuh"
	#include "Physika_Surface_Fuild/SPHsrc/fluid_system_kern.cuh"
#endif
	
	typedef unsigned int		uint;

	struct bufListSWE {
		float3* m_velocity;
		float3* m_midvel;
		float3* m_vector_tmp;
		float3* m_point;
		float3* m_value0;

		float* m_bottom;
		float* m_depth;
		float* m_height;
		float* m_float_tmp;
		float* m_tmp;
		int* m_boundary;

		MyVertex* m_vertex;
		float3 (*m_vertex_rot)[3];
		int (*m_vertex_oneRing)[MAX_VERTEX];
		int (*m_vertex_nearVert)[MAX_NEAR_V];
		int3 (*m_vertex_nerbFace)[MAX_FACES];
		float3 (*m_vertex_planeMap)[MAX_FACES * 3];
		VertexOppositeHalfedge (*m_vertex_opph)[MAX_VERTEX];

		float* m_pressure;

#ifdef HYBRID
		int* m_addLabel; //-1 -> no add
		int* m_addId; //for SPH use
		int* singleValueBuf; ///a small buffer to store temporary single values

		float *sourceTerm;
#endif

		// pull#1: 增加
		bool *m_on_water_boundary; // 在advect末尾更新这个属性
		bool *m_once_have_water;
		float *m_extrapolate_depth; // 在advect末尾更新这个属性。仅在water_boundary计算插值，其它点直接复制depth
		float *m_pressure_gravity;
		float *m_pressure_surface;
		float4 *m_tensor;
		float3 *m_wind_velocity;
		int *m_depth_boundary_type;
		float *m_depth_boundary_value;
	};

	struct FluidParamsSWE {
		int numThreads, numBlocks;
		int vertexNum;
		int szPnts;

		int m_situation;
		float dt;
		float avg_edge_len;
		float m_depth_threshold;
		float3 m_g;

		// pull#1: 增加
		bool m_have_tensor;
		float m_fric_coef;
		float m_gamma;
		float m_water_boundary_theta;
		float m_water_boundary_tension_multiplier;
		float m_max_p_bs;

		float m_wind_coef;
	};

	__global__ void update_mivels_kern ( bufListSWE buf, int vnum );
	__global__ void advect_field_values_kern(bufListSWE buf, int vnum);
	__global__ void update_depth_velocity_kern(bufListSWE buf, int vnum);
	__global__ void extrapolate_depth_kern(bufListSWE buf, int vnum);
	__global__ void force_boundary_depth_kern(bufListSWE buf, int vnum);
	__global__ void compute_pressure_kern(bufListSWE buf, int vnum);
	__global__ void march_water_boundary_pressure_kern(bufListSWE buf, int vnum);
	__global__ void update_velocity_kern ( bufListSWE buf, int vnum );
	__global__ void force_boundary_condition_kern ( bufListSWE buf, int vnum );
	__global__ void velocity_fast_march_kern ( bufListSWE buf, int vnum );
	__global__ void update_velocity_fast_march_kern ( bufListSWE buf, int vnum );
	__global__ void calculate_delta_depth_kern (bufListSWE buf, int vnum);
	__global__ void update_depth_kern ( bufListSWE buf, int vnum );


	void updateSimParams ( FluidParamsSWE* cpufp );

#ifdef HYBRID
	__global__ void addLabelSWE(bufListSWE buf, int vnum);
	__global__ void addParticlesFromSWE(bufListSWE buf, bufList bufSPH, int vnum, int cpnum);
	__global__ void sweInitialAdd();
	__global__ void sweAfterAdd(bufListSWE buf);

	struct GridInfo{
	public:
		float3 gridMin;
		float3 gridDelta;
		int3   gridRes;
		int3   gridScanMax;
		int    gridAdjCnt;
		int    gridAdj[64];

		GridInfo(){ gridMin = make_float3(0, 0, 0); gridDelta = make_float3(0, 0, 0); gridRes = make_int3(0, 0, 0); gridScanMax = make_int3(0, 0, 0);}
		GridInfo(float3 min, float3 delta, int3 res, int3 scan, int adjCnt, int adj[64]){
			gridMin = min; gridDelta = delta; gridRes = res; gridScanMax = scan; 
			gridAdjCnt = adjCnt;
			for (int i = 0; i < adjCnt; i++)
			{
				gridAdj[i] = adj[i];
			}
		}
	};

	__global__ void collectLabelParticles(bufListSWE buf, bufList bufSPH, int vnum, GridInfo gi);

	__global__ void showBug(bufListSWE buf, int vnum);
#endif

#endif
