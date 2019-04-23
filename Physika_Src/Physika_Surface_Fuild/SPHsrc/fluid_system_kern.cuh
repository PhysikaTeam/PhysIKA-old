/*
  FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com

  Attribute-ZLib license (* See additional part 4)

  This software is provided 'as-is', without any express or implied
  warranty. In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
  4. Any published work based on this code must include public acknowledgement
     of the origin. This includes following when applicable:
	   - Journal/Paper publications. Credited by reference to work in text & citation.
	   - Public presentations. Credited in at least one slide.
	   - Distributed Games/Apps. Credited as single line in game or app credit page.	 
	 Retaining this additional license term is required in derivative works.
	 Acknowledgement may be provided as:
	   Publication version:  
	      2012-2013, Hoetzlein, Rama C. Fluids v.3 - A Large-Scale, Open Source
	 	  Fluid Simulator. Published online at: http://fluids3.com
	   Single line (slides or app credits):
	      GPU Fluids: Rama C. Hoetzlein (Fluids v3 2013)

 Notes on Clause 4:
  The intent of this clause is public attribution for this contribution, not code use restriction. 
  Both commerical and open source projects may redistribute and reuse without code release.
  However, clause #1 of ZLib indicates that "you must not claim that you wrote the original software". 
  Clause #4 makes this more specific by requiring public acknowledgement to be extended to 
  derivative licenses. 

*/


#ifndef DEF_KERN_CUDA
	#define DEF_KERN_CUDA

	#include <stdio.h>
	#include <math.h>
	#include "fluid_defs.h"

	typedef unsigned int		uint;
	typedef unsigned short int	ushort;

	struct float9 {
		float			val[9];
	};

	struct lambda {
		float dens;
		float3 sumi;
		float sumj;
	};

	struct mfloat3 {
		float3 tid[TYPE_NUM];
	};

	// Particle & Grid Buffers
	struct bufList {
		float3*			mpos;
		float3*			mvel;
		float3*			mveleval;
		float3*			mforce;
		float*			mpress;
		float*			mdensity;		
		uint*			mgcell;
		uint*			mgndx;
		uint*			mclr;			// 4 byte color

		uint*			mcluster;

		char*			msortbuf;

		uint*			mgrid;	
		int*			mgridcnt;
		int*			mgridoff;
		int*			mgridactive;
		uint*			midsort;

		float3*			mposbak;
		float3*			mdeltapos;
		float*			mlambda;
		float3*			mw;
		float3*			mN;
		float3*			mrestpos;
		float3*			msolidpos;
		float*			mmass;
		float*			mrestdens;
		int*			mtype;			// 0: fluid
		float9*			mQ;
		float3			*A0, *A1, *A2;
		int*			nbcount;
		mfloat*			malpha;    // c: mass fraction
		mfloat3*		galpha;
		mfloat*			deltam;
		mfloat*			mbeta;		// chemical potential

		int*			sweVindex; //vertex index attached to each swe point for SWE particles, or nearest (if != -1) swe point index for fluid particles
		int*			singleValueBuf; ///a small buffer to store temporary single values
	};

	// Temporary sort buffer offsets
	#define BUF_POS			0
	#define BUF_VEL			(sizeof(float3))
	#define BUF_VELEVAL		(BUF_VEL + sizeof(float3))
	#define BUF_FORCE		(BUF_VELEVAL + sizeof(float3))
	#define BUF_PRESS		(BUF_FORCE + sizeof(float3))
	#define BUF_DENS		(BUF_PRESS + sizeof(float))
	#define BUF_GCELL		(BUF_DENS + sizeof(float))
	#define BUF_GNDX		(BUF_GCELL + sizeof(uint))
	#define BUF_CLR			(BUF_GNDX + sizeof(uint))
	#define BUF_POSBAK		(BUF_CLR + sizeof(uint))
	#define BUF_TYPE		(BUF_POSBAK + sizeof(float3))
	#define BUF_MASS		(BUF_TYPE + sizeof(int))
	#define BUF_RESTPOS		(BUF_MASS + sizeof(float))
	#define BUF_RESTDENS	(BUF_RESTPOS + sizeof(float3))
	#define BUF_ALPHA		(BUF_RESTDENS + sizeof(float))
	#define	BUF_SWEVINDEX	(BUF_ALPHA + sizeof(mfloat))  //sweVindex

	// Fluid Parameters (stored on both host and device)
	struct FluidParams {
		int				numThreads, numBlocks;
		int				gridThreads, gridBlocks;	

		int				szPnts, szHash, szGrid;
		int				stride, pnum, snum, fnum, vertnum, facenum, vnnum;
		int				chk;
		float			pdist, pmass, prest_dens;
		mfloat			mpmass, mprest_dens;
		float			pextstiff, pintstiff;
		float			pradius, psmoothradius, r2, psimscale, pvisc;
		float			pforce_min, pforce_max, pforce_freq, pground_slope;
		float			pvel_limit, paccel_limit, pdamp;
		float3			pboundmin, pboundmax, pgravity;
		float			AL, AL2, VL, VL2;

		float			d2, rd2, vterm;		// used in force calculation		 
		
		float			poly6kern, spikykern, lapkern;

		float3			gridSize, gridDelta, gridMin, gridMax;
		int3			gridRes, gridScanMax;
		int				gridSrch, gridTotal, gridAdjCnt, gridActive;

		int				gridAdj[64];
		float			lambda_epsi, vort_epsi;

//#ifdef HYBRID
		int				cfnum; //current fluid particle number
//#endif
	};

	// Prefix Sum defines - 16 banks on G80
	#define NUM_BANKS		16
	#define LOG_NUM_BANKS	 4


	#ifndef CUDA_KERNEL		

		// Declare kernel functions that are available to the host.
		// These are defined in kern.cu, but declared here so host.cu can call them.

		__global__ void insertParticles ( bufList buf, int pnum );
		__global__ void countingSortIndex ( bufList buf, int pnum );		
		__global__ void countingSortFull ( bufList buf, int pnum );	

		__global__ void initialSort ( bufList buf, int pnum );
		__global__ void calcFirstCnt ( bufList buf, int pnum );
		__global__ void countingSortFull_ ( bufList buf, int pnum );
		__global__ void getCnt ( bufList buf, int gnum );


		__global__ void afterSort(bufList buf);
#ifdef HYBRID
		__global__ void afterSWEaddParticle(int pAdded);
		__global__ void prepareLabelParticles(bufList buf, int pnum);
#endif

		__global__ void computeQuery ( bufList buf, int pnum );	
		__global__ void computePressure ( bufList buf, int pnum );		
		__global__ void computeForce ( bufList buf, int pnum );
		__global__ void computePressureGroup ( bufList buf, int pnum );
		__global__ void advanceParticles ( float time, float dt, float ss, bufList buf, int numPnts );

		__global__ void computeExternalForce ( bufList buf, int pnum, float dt, int frame, int ftype );
		__global__ void computeLambda ( bufList buf, int pnum );
		__global__ void computeDeltaPos ( bufList buf, int pnum );
		__global__ void updatePosition ( float time, float dt, bufList buf, int pnum );
		__global__ void updateVelocity ( bufList buf, int pnum, float dt );
		__global__ void vorticityConfinement ( bufList buf, int pnum, float dt );
		__global__ void computeNormal ( bufList buf, int pnum );
		__global__ void computeVorticity ( bufList buf, int pnum );
		__global__ void computeXSPHViscosity ( bufList buf, int pnum );
		__global__ void updateXSPHVelocity ( bufList buf, int pnum );
		__global__ void detectContact ( bufList buf, int num );
		__global__ void computeAQ ( bufList buf, int num, float3 A0, float3 A1, float3 A2 );
		__global__ void computeSolidDeltaPos ( bufList buf, int pnum, float3 mc, float3 restmc );
		__global__ void updateSolidPosition ( bufList buf, int pnum );
		__global__ void calculateA ( bufList buf, int pnum, float3 mc, float3 restmc );
		__global__ void computeChemicalPotential ( bufList buf, int pnum );
		__global__ void computeDeltaM ( bufList buf, int pnum, float dt );
		__global__ void updateM ( bufList buf, int pnum );
		__global__ void computeFlux ( bufList buf, int pnum );
		__global__ void computeGradientC ( bufList buf, int pnum );
		__global__ void computeSF ( bufList buf, int pnum );

		__global__ void updateParticlePosition ( bufList buf, int pnum, int frame );

		__global__ void countActiveCells ( bufList buf, int pnum );		

		void updateSimParams ( FluidParams* cpufp );

		// Prefix Sum
#ifndef HYBRID
		#include "prefix_sum.cu"
		// NOTE: Template functions must be defined in the header
		template <bool storeSum, bool isNP2> __global__ void prescan(float *g_odata, const float *g_idata, float *g_blockSums, int n, int blockIndex, int baseIndex) {
			int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
			extern __shared__ float s_data[];
			loadSharedChunkFromMem<isNP2>(s_data, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
			prescanBlock<storeSum>(s_data, blockIndex, g_blockSums); 
			storeSharedChunkToMem<isNP2>(g_odata, s_data, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
		}
		template <bool storeSum, bool isNP2> __global__ void prescanInt (int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex) {
			int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
			extern __shared__ int s_dataInt [];
			loadSharedChunkFromMemInt <isNP2>(s_dataInt, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
			prescanBlockInt<storeSum>(s_dataInt, blockIndex, g_blockSums); 
			storeSharedChunkToMemInt <isNP2>(g_odata, s_dataInt, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
		}
		__global__ void uniformAddInt (int*  g_data, int *uniforms, int n, int blockOffset, int baseIndex);	
		__global__ void uniformAdd    (float*g_data, float *uniforms, int n, int blockOffset, int baseIndex);	
#endif
	#endif
	

	#define EPSILON				0.00001f
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295
	#define M_PI				3.14159265358979323846

	
#endif
