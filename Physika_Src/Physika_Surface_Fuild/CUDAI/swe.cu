#include <stdio.h>
#include <conio.h>
#include "Physika_Surface_Fuild/Surface_Utilities/cutil.h"
#include "Physika_Surface_Fuild/Surface_Utilities/cutil_math.h"
#include "swe.cuh"
#include "kernel.cuh"



#ifdef HYBRID


extern FluidParams		fcuda;		// CPU Fluid params
extern FluidParams*	mcuda;		// GPU Fluid params

extern bufList			fbuf;		// GPU Particle buffers


#endif

FluidParamsSWE		swcuda;		// CPU Fluid params
FluidParamsSWE*	sgcuda;		// GPU Fluid params

bufListSWE			swbuf;		// GPU Particle buffers

extern bool cudaCheck(cudaError_t status, char* msg);
//bool cudaCheck ( cudaError_t status, char* msg )
//{
//	if (status != cudaSuccess) {
//		printf ( "CUDA ERROR: %s\n", cudaGetErrorString ( status ) );
//		_getch ();
//		return false;
//	}
//	else {
//		//printf ( "%s. OK.\n", msg );
//	}
//	return true;
//}

void cudaInit ( int argc, char **argv )
{
	int count = 0;
	int i = 0;
	cudaError_t err = cudaGetDeviceCount ( &count );
	if (err == cudaErrorInsufficientDriver) 
		printf ( "CUDA driver not installed.\n" );

	if (err == cudaErrorNoDevice)
		printf ( "No CUDA device found.\n" );

	if (count == 0) 
		printf ( "No CUDA device found.\n" );

	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties ( &prop, i ) == cudaSuccess)
			if (prop.major >= 1) break;
	}
	if (i == count) { printf ( "No CUDA device found.\n" ); }
	cudaSetDevice ( i );
	printf ( "CUDA initialized.\n" );

	cudaDeviceProp p;
	cudaGetDeviceProperties ( &p, 0 );

	printf ( "-- CUDA --\n" );
	printf ( "Name:       %s\n", p.name );
	printf ( "Revision:   %d.%d\n", p.major, p.minor );
	printf ( "Global Mem: %d\n", p.totalGlobalMem );
	printf ( "Shared/Blk: %d\n", p.sharedMemPerBlock );
	printf ( "Regs/Blk:   %d\n", p.regsPerBlock );
	printf ( "Warp Size:  %d\n", p.warpSize );
	printf ( "Mem Pitch:  %d\n", p.memPitch );
	printf ( "Thrds/Blk:  %d\n", p.maxThreadsPerBlock );
	printf ( "Const Mem:  %d\n", p.totalConstMem );
	printf ( "Clock Rate: %d\n", p.clockRate );

	// Allocate the sim parameters
	cudaCheck ( cudaMalloc ( (void**)&sgcuda, sizeof ( FluidParamsSWE ) ), "Malloc FluidParamsSWE sgcuda" );
}
void cudaExit ( int argc, char **argv )
{
	cudaDeviceReset ();
}
void cudaFree ()
{
	cudaCheck ( cudaFree ( swbuf.m_bottom ),		"Free bottom" );
	cudaCheck ( cudaFree ( swbuf.m_depth ),		"Free depth" );
	cudaCheck ( cudaFree ( swbuf.m_height ),		"Free height" );
	cudaCheck ( cudaFree ( swbuf.m_float_tmp ),	"Free floattmp" );
	cudaCheck ( cudaFree ( swbuf.m_velocity ),	"Free vel" );
	cudaCheck ( cudaFree ( swbuf.m_midvel ),		"Free midvel" );
	cudaCheck ( cudaFree ( swbuf.m_vector_tmp ), "Free vectortmp" );
	cudaCheck(cudaFree(swbuf.m_vertex), "Free vertex");
	cudaCheck(cudaFree(swbuf.m_vertex_rot), "Free vertex");
	cudaCheck(cudaFree(swbuf.m_vertex_oneRing), "Free vertex");
	cudaCheck(cudaFree(swbuf.m_vertex_nearVert), "Free vertex");
	cudaCheck(cudaFree(swbuf.m_vertex_nerbFace), "Free vertex");
	cudaCheck(cudaFree(swbuf.m_vertex_planeMap), "Free vertex");
	cudaCheck(cudaFree(swbuf.m_vertex_opph), "Free vertex");
	cudaCheck(cudaFree(swbuf.m_pressure), "Free pressure");

#ifdef HYBRID
	cudaCheck(cudaFree(swbuf.m_addLabel), "Free label");
	cudaCheck(cudaFree(swbuf.m_addId), "Free addId");
	cudaCheck(cudaFree(swbuf.singleValueBuf), "Free singleBuf");

	cudaCheck(cudaFree(swbuf.sourceTerm), "Free sourceTerm");
#endif
}

// Compute number of blocks to create
//int iDivUp ( int a, int b ) {
//	return (a % b != 0) ? (a / b + 1) : (a / b);
//}
//void computeNumBlocks ( int numPnts, int maxThreads, int &numBlocks, int &numThreads )
//{
//	numThreads = min ( maxThreads, numPnts );
//	numBlocks = iDivUp ( numPnts, numThreads );
//}
extern int iDivUp(int a, int b);
extern void computeNumBlocks(int numPnts, int maxThreads, int &numBlocks, int &numThreads);

void setupCuda(Simulator const &sim)
{
	//m_mesh.n_vertices(), avg_edge_len, m_depth_threshold, m_dt, m_g[0], m_g[1], m_g[2]
	swcuda.vertexNum = (int)sim.m_mesh.n_vertices();
	swcuda.m_situation = sim.m_situation;
	swcuda.avg_edge_len = sim.avg_edge_len;
	swcuda.m_depth_threshold = sim.m_depth_threshold;
	swcuda.dt = sim.m_dt;
	swcuda.m_g = make_float3(sim.m_g[0], sim.m_g[1], sim.m_g[2]);

	// pull#1: 增加
	swcuda.m_have_tensor = sim.m_have_tensor;
	swcuda.m_fric_coef = sim.m_fric_coef;
	swcuda.m_gamma = sim.m_gamma;
	swcuda.m_water_boundary_theta = sim.m_water_boundary_theta;
	swcuda.m_water_boundary_tension_multiplier = sim.m_water_boundary_tension_multiplier;
	swcuda.m_max_p_bs = sim.m_max_p_bs;
	swcuda.m_wind_coef = sim.m_wind_coef;

	// Compute number of blocks and threads
	int threadsPerBlock = 384;
	computeNumBlocks ( swcuda.vertexNum, threadsPerBlock, swcuda.numBlocks, swcuda.numThreads );				// particles
	// Allocate particle buffers
	swcuda.szPnts = (swcuda.numBlocks  * swcuda.numThreads);

	cudaCheck ( cudaMalloc ( (void**)&swbuf.m_point,		swcuda.szPnts*sizeof ( float3 ) ),	"Malloc point" );
	cudaCheck ( cudaMalloc ( (void**)&swbuf.m_bottom,	swcuda.szPnts*sizeof ( float ) ),	"malloc bottom" );
	cudaCheck ( cudaMalloc ( (void**)&swbuf.m_depth,		swcuda.szPnts*sizeof ( float ) ),	"malloc depth" );
	cudaCheck ( cudaMalloc ( (void**)&swbuf.m_height,	swcuda.szPnts*sizeof ( float ) ),	"malloc height" );
	cudaCheck ( cudaMalloc ( (void**)&swbuf.m_float_tmp, swcuda.szPnts*sizeof ( float ) ),	"malloc floattmp" );
	cudaCheck ( cudaMalloc ( (void**)&swbuf.m_velocity,	swcuda.szPnts*sizeof ( float3 ) ),	"Malloc vel" );
	cudaCheck ( cudaMalloc ( (void**)&swbuf.m_midvel,	swcuda.szPnts*sizeof ( float3 ) ),	"Malloc midvel" );
	cudaCheck ( cudaMalloc ( (void**)&swbuf.m_vector_tmp,swcuda.szPnts*sizeof ( float3 ) ),	"Malloc vectortmp" );
	cudaCheck ( cudaMalloc ( (void**)&swbuf.m_tmp,		MAX_VERTEX*swcuda.szPnts*sizeof ( float ) ), "Malloc vectortmp" );
	cudaCheck ( cudaMalloc ( (void**)&swbuf.m_value0,	swcuda.szPnts*sizeof ( float3 ) ),	"Malloc value0" );
	cudaCheck ( cudaMalloc ( (void**)&swbuf.m_boundary,	swcuda.szPnts*sizeof ( int ) ),		"Malloc boundary" );
	cudaCheck(cudaMalloc((void**)&swbuf.m_vertex, swcuda.szPnts*sizeof (MyVertex)), "Malloc vertex");
	cudaCheck(cudaMalloc((void**)&swbuf.m_vertex_rot, swcuda.szPnts*sizeof (float3[3])), "Malloc vertex");
	cudaCheck(cudaMalloc((void**)&swbuf.m_vertex_oneRing, swcuda.szPnts*sizeof (int[MAX_VERTEX])), "Malloc vertex");
	cudaCheck(cudaMalloc((void**)&swbuf.m_vertex_nearVert, swcuda.szPnts*sizeof (int[MAX_NEAR_V])), "Malloc vertex");
	cudaCheck(cudaMalloc((void**)&swbuf.m_vertex_nerbFace, swcuda.szPnts*sizeof (int3[MAX_FACES])), "Malloc vertex");
	cudaCheck(cudaMalloc((void**)&swbuf.m_vertex_planeMap, swcuda.szPnts*sizeof (float3[MAX_FACES * 3])), "Malloc vertex");
	cudaCheck(cudaMalloc((void**)&swbuf.m_vertex_opph, swcuda.szPnts*sizeof (VertexOppositeHalfedge[MAX_VERTEX])), "Malloc vertex");
	cudaCheck(cudaMalloc((void**)&swbuf.m_pressure, swcuda.szPnts*sizeof (float)), "Malloc pressure");
#ifdef HYBRID
	cudaCheck(cudaMalloc((void**)&swbuf.m_addLabel, swcuda.szPnts*sizeof (int)), "Malloc addLabel");
	cudaCheck(cudaMalloc((void**)&swbuf.m_addId, swcuda.szPnts*sizeof (int)), "Malloc addId");
	cudaCheck(cudaMalloc((void**)&swbuf.singleValueBuf, 10*sizeof (int)), "Malloc singleBuf");

	cudaCheck(cudaMalloc((void**)&swbuf.sourceTerm, swcuda.szPnts*sizeof(float)), "Malloc sourceTerm");
	cudaMemset(swbuf.sourceTerm, 0, swcuda.szPnts*sizeof(float));
#endif
	// pull#1: 增加
	cudaCheck(cudaMalloc((void**)&swbuf.m_on_water_boundary, swcuda.szPnts*sizeof (bool)), "Malloc m_on_water_boundary");
	cudaCheck(cudaMalloc((void**)&swbuf.m_once_have_water, swcuda.szPnts*sizeof (bool)), "Malloc m_once_have_water");
	cudaCheck(cudaMalloc((void**)&swbuf.m_extrapolate_depth, swcuda.szPnts*sizeof (float)), "Malloc m_extrapolate_depth");
	cudaCheck(cudaMalloc((void**)&swbuf.m_pressure_gravity, swcuda.szPnts*sizeof (float)), "Malloc m_pressure_gravity");
	cudaCheck(cudaMalloc((void**)&swbuf.m_pressure_surface, swcuda.szPnts*sizeof (float)), "Malloc m_pressure_surface");
	cudaCheck(cudaMalloc((void**)&swbuf.m_tensor, swcuda.szPnts*sizeof (float4)), "Malloc m_tensor");
	cudaCheck(cudaMalloc((void**)&swbuf.m_wind_velocity, swcuda.szPnts*sizeof (float3)), "Malloc m_wind_velocity");
	cudaCheck(cudaMalloc((void**)&swbuf.m_depth_boundary_type, swcuda.szPnts*sizeof (int)), "Malloc m_depth_boundary_type");
	cudaCheck(cudaMalloc((void**)&swbuf.m_depth_boundary_value, swcuda.szPnts*sizeof (float)), "Malloc m_depth_boundary_value");

	// Transfer sim params to device
	updateSimParams ( &swcuda );

	cudaThreadSynchronize ();
}

void copyToCuda(Simulator const &sim)
{
	int vnum = swcuda.vertexNum;
	cudaCheck(cudaMemcpy(swbuf.m_point, sim.c_point, vnum*sizeof (float3), cudaMemcpyHostToDevice), "Memcpy point ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_bottom, sim.c_bottom, vnum*sizeof (float), cudaMemcpyHostToDevice), "Memcpy bottom toDev");
	cudaCheck(cudaMemcpy(swbuf.m_depth, sim.c_depth, vnum*sizeof (float), cudaMemcpyHostToDevice), "Memcpy depth toDev");
	cudaCheck(cudaMemcpy(swbuf.m_height, sim.c_height, vnum*sizeof (float), cudaMemcpyHostToDevice), "Memcpy height toDev");
	cudaCheck(cudaMemcpy(swbuf.m_velocity, sim.c_vel, vnum*sizeof (float3), cudaMemcpyHostToDevice), "Memcpy vel ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_midvel, sim.c_mvel, vnum*sizeof (float3), cudaMemcpyHostToDevice), "Memcpy midvel ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_boundary, sim.c_boundary, vnum*sizeof (int), cudaMemcpyHostToDevice), "Memcpy height toDev");
	cudaCheck(cudaMemcpy(swbuf.m_value0, sim.c_value0, vnum*sizeof (float3), cudaMemcpyHostToDevice), "Memcpy value0 ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_vertex, sim.c_vertex, vnum*sizeof (MyVertex), cudaMemcpyHostToDevice), "Memcpy vertex ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_vertex_rot, sim.c_vertex_rot, vnum*sizeof (float3[3]), cudaMemcpyHostToDevice), "Memcpy vertex ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_vertex_oneRing, sim.c_vertex_oneRing, vnum*sizeof (int[MAX_VERTEX]), cudaMemcpyHostToDevice), "Memcpy vertex ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_vertex_nearVert, sim.c_vertex_nearVert, vnum*sizeof (int[MAX_NEAR_V]), cudaMemcpyHostToDevice), "Memcpy vertex ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_vertex_nerbFace, sim.c_vertex_nerbFace, vnum*sizeof (int3[MAX_FACES]), cudaMemcpyHostToDevice), "Memcpy vertex ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_vertex_planeMap, sim.c_vertex_planeMap, vnum*sizeof (float3[MAX_FACES * 3]), cudaMemcpyHostToDevice), "Memcpy vertex ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_vertex_opph, sim.c_vertex_opph, vnum*sizeof (VertexOppositeHalfedge[MAX_VERTEX]), cudaMemcpyHostToDevice), "Memcpy vertex ToDev");
	// pull#1: 增加
	cudaCheck(cudaMemcpy(swbuf.m_once_have_water, sim.c_once_have_water, vnum*sizeof (bool), cudaMemcpyHostToDevice), "Memcpy once_have_water toDev");
	cudaCheck(cudaMemcpy(swbuf.m_extrapolate_depth, sim.c_extrapolate_depth, vnum*sizeof (float), cudaMemcpyHostToDevice), "Memcpy extrapolate_depth toDev");
	cudaCheck(cudaMemcpy(swbuf.m_tensor, sim.c_tensor, vnum*sizeof (float4), cudaMemcpyHostToDevice), "Memcpy tensor ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_wind_velocity, sim.c_wind_velocity, vnum*sizeof (float3), cudaMemcpyHostToDevice), "Memcpy wind_velocity ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_depth_boundary_type, sim.c_depth_boundary_type, vnum*sizeof (int), cudaMemcpyHostToDevice), "Memcpy depth_boundary_type ToDev");
	cudaCheck(cudaMemcpy(swbuf.m_depth_boundary_value, sim.c_depth_boundary_value, vnum*sizeof (float), cudaMemcpyHostToDevice), "Memcpy depth_boundary_value ToDev");

	cudaThreadSynchronize ();
}

void copyFromCuda(Simulator const &sim)
{
	int vnum = swcuda.vertexNum;
	//if (sim.c_bottom != 0x0) cudaCheck(cudaMemcpy(sim.c_bottom, swbuf.m_bottom, vnum*sizeof (float), cudaMemcpyDeviceToHost), "Memcpy bottom FromDev");
	//if (sim.c_height != 0x0) cudaCheck(cudaMemcpy(sim.c_height, swbuf.m_height, vnum*sizeof (float), cudaMemcpyDeviceToHost), "Memcpy height FromDev");
	// pull#1: 增加
	if (sim.c_depth != 0x0) cudaCheck(cudaMemcpy(sim.c_depth, swbuf.m_depth, vnum*sizeof (float), cudaMemcpyDeviceToHost), "Memcpy depth FromDev");

	cudaThreadSynchronize ();
}

void debugSyncFromCuda(Simulator &sim) {
	int vnum = swcuda.vertexNum;
	float *pg = new float[vnum];
	float *ps = new float[vnum];
	cudaCheck(cudaMemcpy(sim.c_depth, swbuf.m_depth, vnum*sizeof(float), cudaMemcpyDeviceToHost), "debugSyncFromCuda");
	cudaCheck(cudaMemcpy(sim.c_on_water_boundary, swbuf.m_on_water_boundary, vnum*sizeof(bool), cudaMemcpyDeviceToHost), "debugSyncFromCuda");
	cudaCheck(cudaMemcpy(sim.c_extrapolate_depth, swbuf.m_extrapolate_depth, vnum*sizeof(float), cudaMemcpyDeviceToHost), "debugSyncFromCuda");
	cudaCheck(cudaMemcpy(sim.c_vel, swbuf.m_velocity, vnum*sizeof(float3), cudaMemcpyDeviceToHost), "debugSyncFromCuda");
	cudaCheck(cudaMemcpy(sim.c_mvel, swbuf.m_midvel, vnum*sizeof(float3), cudaMemcpyDeviceToHost), "debugSyncFromCuda");
	cudaCheck(cudaMemcpy(pg, swbuf.m_pressure_gravity, vnum*sizeof(float), cudaMemcpyDeviceToHost), "debugSyncFromCuda");
	cudaCheck(cudaMemcpy(ps, swbuf.m_pressure_surface, vnum*sizeof(float), cudaMemcpyDeviceToHost), "debugSyncFromCuda");
	cudaThreadSynchronize();
	for (int i = 0; i < vnum; i++) {
		MyMesh::VertexHandle vh(i);
		sim.m_mesh.property(sim.m_depth, vh) = sim.c_depth[i];
		sim.m_mesh.property(sim.m_on_water_boundary, vh) = sim.c_on_water_boundary[i];
		sim.m_mesh.property(sim.m_extrapolate_depth, vh) = sim.c_extrapolate_depth[i];
		sim.m_mesh.property(sim.m_velocity, vh) = MyMesh::Point(sim.c_vel[i].x, sim.c_vel[i].y, sim.c_vel[i].z);
		sim.m_mesh.property(sim.m_midvel, vh) = MyMesh::Point(sim.c_mvel[i].x, sim.c_mvel[i].y, sim.c_mvel[i].z);
		sim.m_mesh.property(sim.m_pressure_gravity, vh) = pg[i];
		sim.m_mesh.property(sim.m_pressure_surface, vh) = ps[i];
	}
	delete[] pg;
	delete[] ps;
}

void debugSyncToCuda(Simulator &sim) {
	int vnum = swcuda.vertexNum;
	float *pg = new float[vnum];
	float *ps = new float[vnum];
	for (int i = 0; i < vnum; i++) {
		MyMesh::VertexHandle vh(i);
		sim.c_depth[i] = sim.m_mesh.property(sim.m_depth, vh);
		sim.c_on_water_boundary[i] = sim.m_mesh.property(sim.m_on_water_boundary, vh);
		sim.c_extrapolate_depth[i] = sim.m_mesh.property(sim.m_extrapolate_depth, vh);
		MyMesh::Point tmp = sim.m_mesh.property(sim.m_velocity, vh);
		sim.c_vel[i] = make_float3(tmp[0], tmp[1], tmp[2]);
		tmp = sim.m_mesh.property(sim.m_midvel, vh);
		sim.c_mvel[i] = make_float3(tmp[0], tmp[1], tmp[2]);
		pg[i] = sim.m_mesh.property(sim.m_pressure_gravity, vh);
		ps[i] = sim.m_mesh.property(sim.m_pressure_surface, vh);
	}
	cudaCheck(cudaMemcpy(swbuf.m_depth, sim.c_depth, vnum*sizeof(float), cudaMemcpyHostToDevice), "debugSyncToCuda");
	cudaCheck(cudaMemcpy(swbuf.m_on_water_boundary, sim.c_on_water_boundary, vnum*sizeof(bool), cudaMemcpyHostToDevice), "debugSyncToCuda");
	cudaCheck(cudaMemcpy(swbuf.m_extrapolate_depth, sim.c_extrapolate_depth, vnum*sizeof(float), cudaMemcpyHostToDevice), "debugSyncToCuda");
	cudaCheck(cudaMemcpy(swbuf.m_velocity, sim.c_vel, vnum*sizeof(float3), cudaMemcpyHostToDevice), "debugSyncToCuda");
	cudaCheck(cudaMemcpy(swbuf.m_midvel, sim.c_mvel, vnum*sizeof(float3), cudaMemcpyHostToDevice), "debugSyncToCuda");
	cudaCheck(cudaMemcpy(swbuf.m_pressure_gravity, pg, vnum*sizeof(float), cudaMemcpyHostToDevice), "debugSyncToCuda");
	cudaCheck(cudaMemcpy(swbuf.m_pressure_surface, ps, vnum*sizeof(float), cudaMemcpyHostToDevice), "debugSyncToCuda");
	cudaThreadSynchronize();
	delete[] pg;
	delete[] ps;
	cudaThreadSynchronize();
}

#include "windowstimer.h"
void processCuda(Simulator &sim) {

	WindowsTimer timer;
	auto f = [&]() {
		timer.record();
		return timer.get();
	};
	WindowsTimer::time_t t[8] = { 0, 0, 0, 0, 0, 0 ,0 ,0};
	WindowsTimer::time_t s[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	timer.restart();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: FSInitialAdd1: %s\n", cudaGetErrorString(error));
	}

#if 0
	// 验证oneRing是vv_ccwiter的顺序，验证opph结构的正确性
	int vnum = swcuda.vertexNum;
	MyVertex *mv = new MyVertex[vnum];
	cudaCheck(cudaMemcpy(mv, swbuf.m_vertex, vnum*sizeof(MyVertex), cudaMemcpyDeviceToHost), "vertex");
	for (int i = 0; i < vnum; i++) {
		MyMesh::VertexHandle vh(i);
		int j = 0;
		for (auto vv_it = sim.m_mesh.vv_ccwiter(vh); vv_it.is_valid(); ++vv_it, ++j) {
			if (mv[i].oneRing[j] != vv_it->idx())
				printf("ERROR1 %d %d %d %d\n", i, j, mv[i].oneRing[j], vv_it->idx());
		}
		j = 0;
		for (auto voh_it = sim.m_mesh.voh_ccwiter(vh); voh_it.is_valid(); ++voh_it, ++j) {
			MyVertex::OppositeHalfedge const &opph(mv[i].opph[j]);
			if (!opph.is_valid) printf("ERROR2\n");
			if (opph.is_boundary != sim.m_mesh.is_boundary(*voh_it)) printf("ERROR3\n");
			MyMesh::HalfedgeHandle hh = sim.m_mesh.next_halfedge_handle(*voh_it);
			if (opph.opph_is_boundary != sim.m_mesh.is_boundary(hh)) printf("ERROR4\n");
			if (opph.from_v != sim.m_mesh.from_vertex_handle(hh).idx() || opph.to_v != sim.m_mesh.to_vertex_handle(hh).idx()) {
				printf("ERROR5\n");
				printf("%d %d %d %d\n", opph.from_v, sim.m_mesh.from_vertex_handle(hh).idx(), opph.to_v, sim.m_mesh.to_vertex_handle(hh).idx());
			}
			hh = sim.m_mesh.opposite_halfedge_handle(hh);
			if (opph.opph_is_boundary != sim.m_mesh.is_boundary(hh)) printf("ERROR6\n");
			MyMesh::FaceHandle fh = sim.m_mesh.face_handle(hh);
			//float this_ex_depth = point_interpolation(m_mesh, m_depth, MyMesh::Point(0, 0, 0), *v_it, fh);
		}
	}
	delete[] mv;
#endif
	update_mivels_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();

	advect_field_values_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();

	update_depth_velocity_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();

	extrapolate_depth_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();
	std::swap(swbuf.m_depth, swbuf.m_float_tmp);

	force_boundary_depth_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();

	compute_pressure_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();

	march_water_boundary_pressure_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();

	update_velocity_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();
	
	force_boundary_condition_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();

	velocity_fast_march_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();

	update_velocity_fast_march_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();

	calculate_delta_depth_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();

	update_depth_kern << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
	cudaThreadSynchronize();

	/*
	debugSyncFromCuda(sim);
	sim.update_midvels();
	sim.advect_filed_values();
	sim.extrapolate_depth();
	sim.force_boundary_depth();
	sim.calculate_pressure();
	sim.update_velocity();
	sim.force_boundary_velocity();
	sim.velocity_fast_march();
	sim.update_depth();
	debugSyncToCuda(sim);
	*/
	timer.stop();
}

#ifdef HYBRID
int AddParticlesFromSWE(int currentPnum)
{

	cudaError_t error;

	sweInitialAdd << <1, 1 >> > ();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: FSInitialAdd: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();
	addLabelSWE << <swcuda.numBlocks, swcuda.numThreads >> > (swbuf, swcuda.vertexNum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: FSLabel: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();
	addParticlesFromSWE << <swcuda.numBlocks, swcuda.numThreads >> > (swbuf, fbuf, swcuda.vertexNum, currentPnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: FSAddParticles: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();
	sweAfterAdd << <1, 1 >> >(swbuf);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: FSAfterAdd: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	int pAdded;

	cudaCheck(cudaMemcpy(&pAdded, swbuf.singleValueBuf, sizeof(int), cudaMemcpyDeviceToHost), "Update Particle Number From SWE");
	cudaThreadSynchronize();
	return pAdded;
	//CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&(fcuda_second.pnum), secAddSize, sizeof(int)));  //total pnum changed!
	//CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&(fcuda_second.mf_up), secAddSize, sizeof(int)));  //total pnum changed!
	//computeNumBlocks(fcuda_second.pnum, 384, fcuda_second.numBlocks, fcuda_second.numThreads);    //threads changed!
	//fcuda_second.szPnts = (fcuda_second.numBlocks  * fcuda_second.numThreads);					   //szPnts changed!	
	//error = cudaGetLastError();
	//if (error != cudaSuccess) {
	//	fprintf(stderr, "CUDA ERROR: FSAddNewCUDA: %s\n", cudaGetErrorString(error));
	//}
	//cudaThreadSynchronize();
}

void CollectLabelParticlesSWE()
{
	GridInfo gi(fcuda.gridMin, fcuda.gridDelta, fcuda.gridRes, fcuda.gridScanMax, fcuda.gridAdjCnt, fcuda.gridAdj);

	collectLabelParticles << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, fbuf, swcuda.vertexNum, gi);
	cudaThreadSynchronize();

}
void ShowBug(){
showBug << <swcuda.numBlocks, swcuda.numThreads >> >(swbuf, swcuda.vertexNum);
cudaThreadSynchronize();
}
#endif
