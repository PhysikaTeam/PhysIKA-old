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

#define CUDA_KERNEL
#include "fluid_system_kern.cuh"
#include "fluid_system_host.cuh"

#include "cutil_math.h"

#include "radixsort.cu"						// Build in RadixSort

__device__ FluidParams		simData;
__constant__ uint				gridActive;

__device__ int secAddSize;

__global__ void insertParticles ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	register float3 gridMin = simData.gridMin;
	register float3 gridDelta = simData.gridDelta;
	register int3 gridRes = simData.gridRes;
	register int3 gridScan = simData.gridScanMax;
	register float poff = simData.psmoothradius / simData.psimscale;

	register int		gs;
	register float3		gcf;
	register int3		gc;

	gcf = (buf.mpos[i] - gridMin) * gridDelta; 
	gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
	gs = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;
	if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) {
		buf.mgcell[i] = gs;											// Grid cell insert.
		buf.mgndx[i] = atomicAdd ( &buf.mgridcnt[ gs ], 1 );		// Grid counts.

		gcf = (-make_float3(poff,poff,poff) + buf.mpos[i] - gridMin) * gridDelta;
		gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
		gs = ( gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;		
	} else {
		buf.mgcell[i] = GRID_UNDEF;		
	}
}

// the mutex variable
__device__ int g_mutex = 0;

// GPU simple synchronization function
__device__ void __gpu_sync(int goalVal)
{

	__threadfence ();

	// only thread 0 is used for synchronization
	if (threadIdx.x == 0) 
		atomicAdd(&g_mutex, 1);
	
	// only when all blocks add 1 to g_mutex will
	// g_mutex equal to goalVal
	while(g_mutex < goalVal) {			// infinite loop until g_mutx = goalVal
	}

	if ( blockIdx.x == 0 && threadIdx.x == 0 ) g_mutex = 0;
	
	__syncthreads();
}

// countingSortInPlace -- GPU_SYNC DOES NOT WORK
/*uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) { __gpu_sync ( 2 ); return; }

	register float3	ipos, ivel, iveleval, iforce;
	register float	ipress, idens;
	register int	icell, indx, iclr;

	icell = buf.mgcell [ i ];
	indx = buf.mgndx [ i ];
	int sort_ndx = buf.mgridoff[ icell ] + indx;				// global_ndx = grid_cell_offet + particle_offset
	if ( icell == GRID_UNDEF ) { __gpu_sync ( 2 ); return; }

	ipos = buf.mpos [ i ];
	ivel = buf.mvel [ i ];
	iveleval = buf.mveleval [ i ];
	iforce = buf.mforce [ i ];
	ipress = buf.mpress [ i ];
	idens = buf.mdensity [ i ];
	iclr = buf.mclr [ i ];

	__gpu_sync ( 2 ) ; //threadfence();			// make sure every thread in all blocks has their data

	
	buf.mpos [ sort_ndx ] = ipos;
	buf.mvel [ sort_ndx ] = ivel;
	buf.mveleval [ sort_ndx ] = iveleval;
	buf.mforce [ sort_ndx ] = iforce;
	buf.mpress [ sort_ndx ] = ipress;
	buf.mdensity [ sort_ndx ] = idens;
	buf.mclr [ sort_ndx ] = iclr;*/



// Counting Sort - Index
__global__ void countingSortIndex ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) return;

	uint icell = buf.mgcell[i];
	uint indx =  buf.mgndx[i];
	//int sort_ndx = buf.mgridoff[ icell ] + indx;				// global_ndx = grid_cell_offet + particle_offset
	if ( icell != GRID_UNDEF ) {
		int sort_ndx = buf.mgridoff[icell] + indx;				// global_ndx = grid_cell_offet + particle_offset
		buf.mgrid[ sort_ndx ] = i;					// index sort, grid refers to original particle order
	}
}

// Counting Sort - Full (deep copy)
__global__ void countingSortFull ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) return;

	// Copy particle from original, unsorted buffer (msortbuf),
	// into sorted memory location on device (mpos/mvel)
	uint icell = *(uint*) (buf.msortbuf + pnum*BUF_GCELL + i*sizeof(uint) );
	uint indx =  *(uint*) (buf.msortbuf + pnum*BUF_GNDX + i*sizeof(uint) );		

	if ( icell != GRID_UNDEF ) {	  
		// Determine the sort_ndx, location of the particle after sort
	    int sort_ndx = buf.mgridoff[ icell ] + indx;				// global_ndx = grid_cell_offet + particle_offset	
		
		// Find the original particle data, offset into unsorted buffer (msortbuf)
		char* bpos = buf.msortbuf + i*sizeof(float3);

		// Transfer data to sort location
		buf.mgrid[ sort_ndx ] = sort_ndx;			// full sort, grid indexing becomes identity		
		buf.mpos[ sort_ndx ] =		*(float3*) (bpos);
		buf.mvel[ sort_ndx ] =		*(float3*) (bpos + pnum*BUF_VEL );
		buf.mveleval[ sort_ndx ] =	*(float3*) (bpos + pnum*BUF_VELEVAL );
		buf.mforce[ sort_ndx ] =	*(float3*) (bpos + pnum*BUF_FORCE );
		buf.mpress[ sort_ndx ] =	*(float*) (buf.msortbuf + pnum*BUF_PRESS + i*sizeof(float) );
		buf.mdensity[ sort_ndx ] =	*(float*) (buf.msortbuf + pnum*BUF_DENS + i*sizeof(float) );
		buf.mclr[ sort_ndx ] =		*(uint*) (buf.msortbuf + pnum*BUF_CLR + i*sizeof(uint) );		// ((uint) 255)<<24; -- dark matter
		buf.mposbak[ sort_ndx ] =	*(float3*) (buf.msortbuf + pnum*BUF_POSBAK + i*sizeof(float3) );
		buf.mtype[ sort_ndx ] =		*(int*) (buf.msortbuf + pnum*BUF_TYPE + i*sizeof ( int ));
		buf.mmass[ sort_ndx ] =		*(float*) (buf.msortbuf + pnum*BUF_MASS + i*sizeof ( float ));
		buf.mrestpos[ sort_ndx ] =	*(float3*) (buf.msortbuf + pnum*BUF_RESTPOS + i*sizeof ( float3 ));
		buf.mrestdens[ sort_ndx ] = *(float*)(buf.msortbuf + pnum*BUF_RESTDENS + i*sizeof ( float ));
		buf.malpha[ sort_ndx ] =	*(mfloat*)(buf.msortbuf + pnum*BUF_ALPHA + i*sizeof ( mfloat ));

		buf.sweVindex[sort_ndx] = *(int*)(buf.msortbuf + pnum*BUF_SWEVINDEX + i*sizeof(int));

		buf.mgcell[ sort_ndx ] =	icell;
		buf.mgndx[ sort_ndx ] =		indx;		
	}
}

__global__ void initialSort ( bufList buf, int pnum )
{

	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	register float3 gridMin = simData.gridMin;
	register float3 gridDelta = simData.gridDelta;
	register int3 gridRes = simData.gridRes;
	register int3 gridScan = simData.gridScanMax;

	register int		gs;
	register float3		gcf;
	register int3		gc;

	gcf = (buf.mpos[i] - gridMin) * gridDelta;
	gc = make_int3 ( int ( gcf.x ), int ( gcf.y ), int ( gcf.z ) );
	gs = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;
	if (gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z) {
		buf.mgcell[i] = gs;											// Grid cell insert.
		buf.midsort[i] = i;
		//		buf.mgndx[i] = atomicAdd ( &buf.mgridcnt[ gs ], 1 );		// Grid counts.
		//		gcf = (-make_float3(poff,poff,poff) + buf.mpos[i] - gridMin) * gridDelta;
		//		gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
		//		gs = ( gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;
		//buf.mcluster[i] = gs;				-- make sure it is allocated!
	}
	else {
		buf.mgcell[i] = GRID_UNDEF;
		buf.midsort[i] = i;
		//buf.mcluster[i] = GRID_UNDEF;		-- make sure it is allocated!
		//if (buf.mtype[i] == -2) buf.mgcell[i] = GRID_UNDEF - 1;
	}
}



// ***** UNUSED CODE (not working) ******
__global__ void countActiveCells ( bufList buf, int pnum )
{	
	if ( threadIdx.x == 0 ) {		
		// use only one processor
		
		//gridActive = -1;

		int last_ndx = buf.mgridoff [ simData.gridTotal-1 ] + buf.mgridcnt[ simData.gridTotal-1 ] - 1;
		int last_p = buf.mgrid[ last_ndx ];
		int last_cell = buf.mgcell[ last_p ];
		int first_p = buf.mgrid[ 0 ];
		int first_cell = buf.mgcell[ first_p ] ;

		int cell, cnt = 0, curr = 0;
		cell = first_cell;
		while ( cell < last_cell ) {			
			buf.mgridactive[ cnt ] = cell;			// add cell to active list
			cnt++;
			curr += buf.mgridcnt[cell];				// advance to next active cell
			// id = buf.mgrid[curr];				// get particle id -- when unsorted only
			cell = buf.mgcell [ curr ];				// get cell we are in -- use id when unsorted
		}
		// gridActive = cnt;
	}
	__syncthreads();
}

// Calculate first and cnt for each grid
__global__ void calcFirstCnt ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	if ((i == 0 || buf.mgcell[i] != buf.mgcell[i - 1]))
	{
		if (buf.mgcell[i] != GRID_UNDEF)buf.mgridoff[buf.mgcell[i]] = i;
	}
	__syncthreads ();
	if (i != 0 && buf.mgcell[i] != buf.mgcell[i - 1] && buf.mgcell[i - 1] != GRID_UNDEF)
		buf.mgridcnt[buf.mgcell[i - 1]] = i;
	if (i == pnum - 1 && buf.mgcell[i] != GRID_UNDEF)
		buf.mgridcnt[buf.mgcell[i]] = i + 1;

//#ifdef HYBRID
	if (buf.mgcell[i] == GRID_UNDEF && (i == 0 || buf.mgcell[i - 1] != GRID_UNDEF))
	{
		simData.cfnum = i;
	}
//#endif
}

__global__ void countingSortFull_ ( bufList buf, int pnum )
{

	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;		// particle index				
	if (i >= pnum) return;

	uint icell = *(uint*)(buf.msortbuf + pnum*BUF_GCELL + i*sizeof ( uint ));
	uint indx = *(uint*)(buf.msortbuf + pnum*BUF_GNDX + i*sizeof ( uint ));
	//uint j = i;
	i = buf.midsort[i];				// ?
	if (icell != GRID_UNDEF) {
		int sort_ndx = buf.mgridoff[icell] + indx;				// global_ndx = grid_cell_offet + particle_offset
		buf.mgrid[sort_ndx] = sort_ndx;			// full sort, grid indexing becomes identity
		char* bpos = buf.msortbuf + i*sizeof ( float3 );
		buf.mpos[ sort_ndx ] =		*(float3*)(bpos);
		buf.mvel[ sort_ndx ] =		*(float3*)(bpos + pnum*BUF_VEL);
		buf.mveleval[ sort_ndx ] =	*(float3*)(bpos + pnum*BUF_VELEVAL);
		buf.mforce[ sort_ndx ] =	*(float3*)(bpos + pnum*BUF_FORCE);
		buf.mpress[ sort_ndx ] =	*(float*)(buf.msortbuf + pnum*BUF_PRESS + i*sizeof ( float ));
		buf.mdensity[ sort_ndx ] =	*(float*)(buf.msortbuf + pnum*BUF_DENS + i*sizeof ( float ));
		buf.mclr[ sort_ndx ] =		*(uint*)(buf.msortbuf + pnum*BUF_CLR + i*sizeof ( uint ));		// ((uint) 255)<<24; -- dark matter
		buf.mposbak[ sort_ndx ] =	*(float3*)(buf.msortbuf + pnum*BUF_POSBAK + i*sizeof ( float3 ));
		buf.mtype[ sort_ndx ] =		*(int*)(buf.msortbuf + pnum*BUF_TYPE + i*sizeof ( int ));
		buf.mmass[ sort_ndx ] =		*(float*)(buf.msortbuf + pnum*BUF_MASS + i*sizeof ( float ));
		buf.mrestpos[ sort_ndx ] =	*(float3*)(buf.msortbuf + pnum*BUF_RESTPOS + i*sizeof ( float3 ));
		buf.mrestdens[ sort_ndx ] = *(float*)(buf.msortbuf + pnum*BUF_RESTDENS + i*sizeof ( float ));
		buf.malpha[ sort_ndx ] =	*(mfloat*)(buf.msortbuf + pnum*BUF_ALPHA + i*sizeof ( mfloat ));

		buf.sweVindex[sort_ndx] = *(int*)(buf.msortbuf + pnum*BUF_SWEVINDEX + i*sizeof(int));

		buf.mgcell[ sort_ndx ] = icell;
		buf.mgndx[ sort_ndx ] = indx;
	}
	else
	{
		//		buf.mgcell[ sort_ndx ] =	GRID_UNDEF;
		////buf.mpos[j] = make_float3(0,-10,0);
		////buf.mvel[j] = make_float3(0,0,0);
	}
}

__global__ void getCnt ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;		// particle index
	if (i >= pnum) return;
	if (buf.mgcell[i] != GRID_UNDEF)
	{
		buf.mgndx[i] = i - buf.mgridoff[buf.mgcell[i]];
		if (buf.mgndx[i] == 0)
			buf.mgridcnt[buf.mgcell[i]] -= buf.mgridoff[buf.mgcell[i]];
	}
}

__global__ void afterSort(bufList buf)
{
	simData.fnum = simData.cfnum;
	simData.pnum = simData.cfnum + simData.snum;
	buf.singleValueBuf[0] = simData.cfnum;

}
#ifdef HYBRID
__global__ void afterSWEaddParticle(int pAdded)
{
	simData.cfnum += pAdded;
	simData.fnum = simData.cfnum;
	simData.pnum = simData.fnum + simData.snum;
}
#endif

__device__ float contributePressure ( int i, float3 p, int cell, bufList buf )
{			
	float3 dist;
	float dsq, c, sum;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2 / d2;
	float s = 1.5;
	//if (buf.mtype[i] != 0) s = 2.0; // ?
	sum = 0.0;

	if ( buf.mgridcnt[cell] == 0 ) return 0.0;
	
	int cfirst = buf.mgridoff[ cell ];
	int clast = cfirst + buf.mgridcnt[ cell ];
	
	for ( int cndx = cfirst; cndx < clast; cndx++ ) {
		uint j = buf.mgrid[cndx];
#ifdef HYBRID
		if (buf.mtype[j] == -3)
			continue;
#endif
		dist = p - buf.mpos[ j ];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if ( dsq < r2 ){ // && dsq > 0.0) {
			c = (r2 - dsq)*d2;
#ifndef HYBRIDSPHSIM
			if (buf.mtype[j] == 0)
				sum += c * c * c * buf.mmass[i]; // mi or mj
			else if (buf.mtype[j] < 0)
				sum += c * c * c * buf.mmass[j];
			else
				sum += s * c * c * c * buf.mmass[i]; // mi or mj
#else
		sum += c*c*c*buf.mmass[j];
		buf.nbcount[i]++;
#endif
			//buf.nbcount[i]++; // count of neighbor particles
		} 
	}
	
	return sum;
}
			
__global__ void computePressure ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;
	if (buf.mtype[i] < 0) return;
	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;

	// Sum Pressures
	float3 pos = buf.mpos[ i ];
	float sum = 0.0;
	buf.nbcount[i] = 0;
	for (int c=0; c < simData.gridAdjCnt; c++) {
		sum += contributePressure ( i, pos, gc + simData.gridAdj[c], buf );
	}
	//__syncthreads(); // ?
		
	// Compute Density & Pressure
	sum = sum /** simData.pmass*/ * simData.poly6kern;
	//if ( sum == 0.0 ) sum = 1.0;
	sum = max ( 1e-6, sum );
#ifdef HYBRIDSPHSIM
	float dens = buf.mrestdens[i];
	//buf.mpress[i] = (sum - /*simData.prest_dens*/buf.mrestdens[i]) * simData.pintstiff;
	buf.mpress[i] = simData.pintstiff * dens * (pow(sum/dens, 7.0f)-1);
	buf.mdensity[i] = 1.0 / sum;

	if (i ==23210)
	{
		printf("%d: %f, %d, %f, %f\n", i, sum, buf.nbcount[i], buf.mmass[i]*10000, simData.poly6kern);
	}
#else
	buf.mdensity[i] = sum; // 1.0f / sum;
#endif
}

		
__global__ void computeQuery ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;

	// Sum Pressures
	float3 pos = buf.mpos[ i ];
	float sum = 0.0;
	for (int c=0; c < simData.gridAdjCnt; c++) {
		sum += 1.0;
	}
	__syncthreads();
	
}

/*FindNeighbors
int cid = blockIdx.x * blockSize.x + blockIdx.y;   // cluster id	
int pid = threadIdx.x;		           // 0 to 85 (max particles per cell)	
__shared__ Particle  clist[ 85 ];	
__shared__ Particle  plist[ 85*8 ];
if ( pid < clusterCnt[cid] )  
	clist [ pid ] = particles [ clusterNdx[cid] + pid ];

for ( gid = 0;  gid < 8;  gid++ ) {
	if ( pid < gridCnt[  cid + group[gid] ] )  
		plist [ cid*CELL_CNT + pid ] = particles [ sortNdx[ cid + group[gid] ]  + pid ]; 	}

__syncthreads();	
	
for ( int j = 0; j < cellcnt;  j++ ) {
	dst = plist[ pid ] - plist[ j ];
	if ( dst < R2 ) {
     		  ...
	}
}*/

/*grid		    block
<gx, gy, gz>    <1, 32, 64>
256, 256, 256  
total:  */


#define LOCAL_PMAX		896
#define NUM_CELL		27
#define LAST_CELL		26
#define CENTER_CELL		13

__global__ void computePressureGroup ( bufList buf, int pnum )
{
	__shared__ float3	cpos[ LOCAL_PMAX ];

	__shared__ int		ncnt[ NUM_CELL ];
	__shared__ int		ngridoff[ NUM_CELL ];
	__shared__ int		noff[ NUM_CELL ];
	
	int bid = __mul24( blockIdx.y, gridDim.x ) + blockIdx.x;
	if ( bid > gridActive ) return;				// block must be in a valid grid
	uint cell = buf.mgridactive [ bid ];		// get grid cell (from blockID 1:1)
	register int i = -1;
	register float3 ipos;

	uint ndx = threadIdx.x;							
	if ( ndx < buf.mgridcnt[cell] ) {
		i = buf.mgridoff[cell] + ndx;		// particle id to process
		ipos = buf.mpos[ i ];
	}
	int gid = threadIdx.x;

	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2 / d2;
	register float3 dist;
	register float c, dsq, sum;
	int neighbor;

	// copy neighbor cell counts to shared mem
	if ( gid < NUM_CELL ) {
		int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
		neighbor = cell - nadj + simData.gridAdj[gid];					// neighbor cell id
		ncnt[gid] = buf.mgridcnt [ neighbor ];	
		ngridoff[gid] = buf.mgridoff [ neighbor ];
	}
	__syncthreads ();

	if ( gid == 0 ) {									// compute neighbor local ndx (as prefix sum)
		int nsum = 0;
		for (int z=0; z < NUM_CELL; z++) {				// 27-step prefix sum
			noff[z] = nsum;
			nsum += ncnt[z];
		}
	}
	__syncthreads ();

	// copy particles into shared memory
	if ( gid < NUM_CELL ) {
		for (int j=0; j < ncnt[gid]; j++ ) {
			neighbor = buf.mgrid [ ngridoff[gid] + j ];		// neighbor particle id
			ndx = noff[ gid ] + j;
			cpos[ ndx ] = buf.mpos [ neighbor ];
		}
	}
	__syncthreads ();

	
	// compute pressure for current particle
	if ( i == -1 ) return;
	
	int jnum = noff[LAST_CELL] + ncnt[LAST_CELL];
	sum = 0.0;
	for (int j = 0; j < jnum; j++) {
		dist = ipos - cpos[ j ];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);			
		if ( dsq > 0.0 && dsq < r2 ) {
			c = (r2 - dsq)*d2;
			sum += c * c * c;
		}
	}	
	__syncthreads ();

	// put result into global mem
	sum = sum * simData.pmass * simData.poly6kern;
	if ( sum == 0.0 ) sum = 1.0;
	buf.mpress[ i ] = ( sum - simData.prest_dens ) * simData.pintstiff;
	buf.mdensity[ i ] = 1.0f / sum; 	
}


__device__ float3 contributeForce ( int i, float3 ipos, float3 iveleval, float ipress, float idens, int cell, bufList buf )
{			
	float dsq, c;	
	float pterm;
	float3 dist, force;	
	int j;					

	if ( buf.mgridcnt[cell] == 0 ) return make_float3(0,0,0);	

	force = make_float3(0,0,0);

	for ( int cndx = buf.mgridoff[ cell ]; cndx < buf.mgridoff[ cell ] + buf.mgridcnt[ cell ]; cndx++ ) {										
		j = buf.mgrid[ cndx ];			
#ifdef HYBRID
		if (buf.mtype[j] == -3)
			continue;
#endif

		dist = ( ipos - buf.mpos[ j ] );		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		//dist *= simData.psimscale;
		if ( dsq < simData.rd2 && dsq > 0) {			
			dsq = sqrt(dsq * simData.d2);
			c = ( simData.psmoothradius - dsq ); 
			pterm = simData.psimscale * -0.5f * c * simData.spikykern * ( ipress + buf.mpress[ j ] ) / dsq;			
#ifndef HYBRIDSPHSIM
			force += ( pterm * dist + simData.vterm * ( buf.mveleval[ j ] - iveleval )) * c * idens * (buf.mdensity[ j ] ); //original
#else
			force += (pterm * dist + simData.vterm * (buf.mveleval[j] - iveleval)) * c * idens * (buf.mdensity[j]) * buf.mmass[j]; //original
#endif
			//if (i == 21704)
			//{
			//	printf("A:%f, %f, %f, %f\n", simData.spikykern, simData.psimscale, dsq, pterm);
			//	printf("B:%f, %f, %f, %f\n", dist.x, c, buf.mdensity[j], buf.mmass[j] * 10000);
			//	printf("C:%f, %f, %f\n", idens, ipos.x, buf.mpos[j].x);
			//}
		}	
	}
	return force;
}


__global__ void computeForce ( bufList buf, int pnum)
{			
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;
	if (buf.mtype[i] < 0) return;
	// Get search cell	
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;

	// Sum Pressures	
	register float3 force;
	force = make_float3(0,0,0);		

	for (int c=0; c < simData.gridAdjCnt; c++) {
		force += contributeForce ( i, buf.mpos[ i ], buf.mveleval[ i ], buf.mpress[ i ], buf.mdensity[ i ], gc + simData.gridAdj[c], buf );
	}
	//__syncthreads ();
	buf.mforce[ i ] = force;
	//printf("%d, Force:%f,%f,%f\n",i, force.x, force.y, force.z);
	if (i == 23210)
	{
		printf("force: %f, %f, %f\n", force.x, force.y, force.z);
	}
	//if (i % 1000 == 0) printf ( "force = %f\n", length(force) * simData.pmass );
}
	

/*__global__ void computeForceNbr ( char* bufPnts, int* bufGrid, int numPnt )
{		
	uint ndx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index		
	if ( ndx >= numPnt ) return;
				
	char* ioffs = bufPnts + __mul24(ndx, simData.stride );
	float3 ipos = *(float3*)	(ioffs + OFFSET_POS);
	float3 ivelval = *(float3*)	(ioffs + OFFSET_VELEVAL);
	float press = *(float*)		(ioffs + OFFSET_PRESS);
	float dens =  *(float*)		(ioffs + OFFSET_DENS);
	int icnt =  *(int*)			(ioffs + OFFSET_NBRCNT);

	char* joffs;
	float3 jpos, jveleval;

	float3 dist, force;		
	float c, ndistj, pterm, dterm, vterm;
		
	vterm = simData.lapkern * simData.visc;
		
	force = make_float3(0,0,0);
	for (int nbr=0; nbr < icnt; nbr++) {		// base 1, n[0] = count
		ndistj = bufNdist[ndx][nbr];
		joffs = bufPnts + __mul24(bufNeighbor[ndx][nbr], simData.stride);
		jpos = *(float3*)		(joffs + OFFSET_POS);
		jveleval = *(float3*)	(joffs + OFFSET_VELEVAL);
		c = ( simData.smooth_rad - ndistj ); 
		dist.x = ( ipos.x - jpos.x );		// dist in cm
		dist.y = ( ipos.y - jpos.y );
		dist.z = ( ipos.z - jpos.z );			
		pterm = simData.sim_scale * -0.5f * c * simData.spikykern * ( press + *(float*)(joffs+OFFSET_PRESS) ) / ndistj;
		dterm = c * dens * *(float*)(joffs+OFFSET_DENS);	
		force.x += ( pterm * dist.x + vterm * ( jveleval.x - ivelval.x )) * dterm;
		force.y += ( pterm * dist.y + vterm * ( jveleval.y - ivelval.y )) * dterm;
		force.z += ( pterm * dist.z + vterm * ( jveleval.z - ivelval.z )) * dterm;			
	}
	*(float3*) ( ioffs + OFFSET_FORCE ) = force;		
}*/

		
__global__ void advanceParticles ( float time, float dt, float ss, bufList buf, int numPnts )
{		
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;
	if (buf.mtype[i] < 0) return;

	if ( buf.mgcell[i] == GRID_UNDEF ) {
		buf.mpos[i] = make_float3(-1000,-1000,-1000);
		buf.mvel[i] = make_float3(0,0,0);
		return;
	}
			
	// Get particle vars
	register float3 accel, norm;
	register float diff, adj, speed;
	register float3 pos = buf.mpos[i];
	register float3 veval = buf.mveleval[i];

	// Leapfrog integration						
	accel = buf.mforce[i];
#ifndef HYBRIDSPHSIM
	accel *= simData.pmass;
#endif
	// Boundaries
	// Y-axis

//#ifndef HYBRIDSPHSIM
	//diff = simData.pradius - (pos.y - (simData.pboundmin.y + (pos.x-simData.pboundmin.x)*simData.pground_slope )) * ss;
	diff = -(pos.y - (-25))*ss;
	if ( diff > EPSILON ) {
		//norm = make_float3( -simData.pground_slope, 1.0 - simData.pground_slope, 0);
		norm = make_float3(0, 1, 0);
		adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	diff =  - ( 50 - pos.y )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3(0, -1, 0);
		adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	// X-axis
	//diff = simData.pradius - (pos.x - (simData.pboundmin.x + (sin(time*simData.pforce_freq)+1)*0.5 * simData.pforce_min))*ss;
	diff = -(pos.x - (0))*ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 1, 0, 0);
		adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
	//diff = simData.pradius - ( (simData.pboundmax.x - (sin(time*simData.pforce_freq)+1)*0.5*simData.pforce_max) - pos.x)*ss;
	diff = -(50 - pos.x)*ss;
	if ( diff > EPSILON ) {
		norm = make_float3(-1, 0, 0);
		adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	// Z-axis
	//diff = simData.pradius - (pos.z - simData.pboundmin.z ) * ss;
	diff = -(pos.z - (0)) * ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, 1 );
		adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
	//diff = simData.pradius - ( simData.pboundmax.z - pos.z )*ss;
	diff = -(50 - pos.z)*ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, -1 );
		adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
//#endif


	// Accel Limit
	speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
	if ( speed > simData.AL2 ) {
		accel *= simData.AL / sqrt(speed);
	}

	// Gravity
	accel += simData.pgravity;

	float3 vel = buf.mvel[i];
#ifndef HYBRIDSPHSIM
	// Velocity Limit
	
	speed = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
	if ( speed > simData.VL2 ) {
		speed = simData.VL2;
		vel *= simData.VL / sqrt(speed);
	}


	// Ocean colors
	if ( speed > simData.VL2*0.2) {
		adj = simData.VL2*0.2;
		buf.mclr[i] += ((  buf.mclr[i] & 0xFF) < 0xFD ) ? +0x00000002 : 0;		// decrement R by one
		buf.mclr[i] += (( (buf.mclr[i]>>8) & 0xFF) < 0xFD ) ? +0x00000200 : 0;	// decrement G by one
		buf.mclr[i] += (( (buf.mclr[i]>>16) & 0xFF) < 0xFD ) ? +0x00020000 : 0;	// decrement G by one
	}
	if ( speed < 0.03 ) {		
		int v = int(speed/.01)+1;
		buf.mclr[i] += ((  buf.mclr[i] & 0xFF) > 0x80 ) ? -0x00000001 * v : 0;		// decrement R by one
		buf.mclr[i] += (( (buf.mclr[i]>>8) & 0xFF) > 0x80 ) ? -0x00000100 * v : 0;	// decrement G by one
	}
	
	//-- surface particle density 
	//buf.mclr[i] = buf.mclr[i] & 0x00FFFFFF;
	//if ( buf.mdensity[i] > 0.0014 ) buf.mclr[i] += 0xAA000000;
#endif
	// Leap-frog Integration
	//buf.mclr[i] = COLORA((buf.mrestdens[i] == 0.35f) ? 0 : 1, (buf.mrestdens[i] == 0.35f) ? 1 : 0, 0, 1);
	//if (i == 21704)printf("%d: %f, %f, FORCE(%f,%f,%f)\n", i, buf.mpress[i], simData.AL2, buf.mforce[i].x, buf.mforce[i].y, buf.mforce[i].z);

	float3 vnext = accel*dt + vel;				// v(t+1/2) = v(t-1/2) + a(t) dt		
	buf.mveleval[i] = (vel + vnext) * 0.5;		// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
	buf.mvel[i] = vnext;
	buf.mpos[i] += vnext * (dt/ss);						// p(t+1) = p(t) + v(t+1/2) dt		

	//if (buf.mpos[i].x > 0)buf.mpos[i].x = 0;
	//if (buf.mpos[i].x < -30)buf.mpos[i].x = -30;
	//if (buf.mpos[i].z > 30)buf.mpos[i].z = 30;
	//if (buf.mpos[i].z < -30)buf.mpos[i].z = -30;
	//if (buf.mpos[i].y < 10)buf.mpos[i].y = 10;
}

__global__ void computeExternalForce ( bufList buf, int pnum, float dt, int frame, int ftype )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	buf.mrestpos[i] = make_float3 ( 0, 0, 0 );
	buf.mposbak[i] = buf.mpos[i];
	if (buf.mtype[i] < 0) return;

	if (buf.mtype[i] == 0) buf.mrestpos[i] = make_float3 ( 0, 0, 0 );
	else buf.mrestpos[i] = buf.mpos[i];
	//buf.mposbak[i] = buf.mpos[i];
	buf.mveleval[i] = buf.mvel[i];
	buf.mvel[i] += dt * (simData.pgravity); // external force

	/*switch (ftype) {
	case 1: if (buf.mpos[i].x < 0) buf.mvel[i] += dt * make_float3 ( 100, 0, 0 ); break;
	case 2: if (buf.mpos[i].x > 0) buf.mvel[i] += dt * make_float3 ( -100, 0, 0 ); break;
	case 3: if (buf.mpos[i].z < 0) buf.mvel[i] += dt * make_float3 ( 0, 0, 100 ); break;
	case 4: if (buf.mpos[i].z > 0) buf.mvel[i] += dt * make_float3 ( 0, 0, -100 ); break;
	default: break;
	}*/

	buf.mpos[i] += (dt) * buf.mvel[i];
}

__global__ void detectContact ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	buf.msolidpos[i] = make_float3 ( 0, 0, 0 );
	if (buf.mtype[i] <= 0) return;

	float ss = simData.psimscale;
	float radius = 0 * simData.pradius / ss;

	buf.mpos[i] = clamp ( buf.mpos[i], simData.pboundmin + radius, simData.pboundmax - radius );
	buf.msolidpos[i] = buf.mpos[i];
}

__global__ void calculateA ( bufList buf, int pnum, float3 mc, float3 restmc )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	buf.A0[i] = buf.A1[i] = buf.A2[i] = make_float3 ( 0, 0, 0 );
	if (buf.mtype[i] <= 0) return;

	float3 a = buf.mpos[i] - mc;
	float3 b = buf.mrestpos[i] - restmc;
	buf.A0[i].x = a.x*b.x; buf.A0[i].y = a.x*b.y; buf.A0[i].z = a.x*b.z;
	buf.A1[i].x = a.y*b.x; buf.A1[i].y = a.y*b.y; buf.A1[i].z = a.y*b.z;
	buf.A2[i].x = a.z*b.x; buf.A2[i].y = a.z*b.y; buf.A2[i].z = a.z*b.z;
}

__device__ float oneNorm ( const float * A )
{
	float norm = 0.0;
	for (int i = 0; i<3; i++)
	{
		float columnAbsSum = fabs ( A[i + 0] ) + fabs ( A[i + 3] ) + fabs ( A[i + 6] );
		if (columnAbsSum > norm)
			norm = columnAbsSum;
	}
	return norm;
}

__device__ float infNorm ( const float * A )
{
	float norm = 0.0;
	for (int i = 0; i<3; i++)
	{
		float rowSum = fabs ( A[3 * i + 0] ) + fabs ( A[3 * i + 1] ) + fabs ( A[3 * i + 2] );
		if (rowSum > norm)
			norm = rowSum;
	}
	return norm;
}

__device__ void crossProduct ( const float * a, const float * b, float * c )
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

__global__ void computeAQ ( bufList buf, int pum, float3 A0, float3 A1, float3 A2 )
{
	float9 A, Q;
	A.val[0] = A0.x; A.val[1] = A0.y; A.val[2] = A0.z;
	A.val[3] = A1.x; A.val[4] = A1.y; A.val[5] = A1.z;
	A.val[6] = A2.x; A.val[7] = A2.y; A.val[8] = A2.z;
	for (int i = 0; i < 9; i++) {
		Q.val[i] = 0;
	}
	float eps = 1e-6;
	float S[9], Ak[9], Ek[9];
	float det, M_oneNorm, M_infNorm, E_oneNorm;

	// Ak = A^T
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			Ak[3 * i + j] = A.val[3 * j + i];

	M_oneNorm = oneNorm ( Ak );
	M_infNorm = infNorm ( Ak );

	do
	{
		float MadjTk[9];

		// row 2 x row 3
		crossProduct ( &(Ak[3]), &(Ak[6]), &(MadjTk[0]) );
		// row 3 x row 1
		crossProduct ( &(Ak[6]), &(Ak[0]), &(MadjTk[3]) );
		// row 1 x row 2
		crossProduct ( &(Ak[0]), &(Ak[3]), &(MadjTk[6]) );

		det = Ak[0] * MadjTk[0] + Ak[1] * MadjTk[1] + Ak[2] * MadjTk[2];
		if (det == 0.0)
		{
			printf ( "Warning (polarDecomposition) : zero determinant encountered.\n" );
			break;
		}

		float MadjT_one = oneNorm ( MadjTk );
		float MadjT_inf = infNorm ( MadjTk );

		float gamma = sqrt ( sqrt ( (MadjT_one * MadjT_inf) / (M_oneNorm * M_infNorm) ) / fabs ( det ) );
		float g1 = gamma * 0.5;
		float g2 = 0.5 / (gamma * det);

		for (int i = 0; i<9; i++)
		{
			Ek[i] = Ak[i];
			Ak[i] = g1 * Ak[i] + g2 * MadjTk[i];
			Ek[i] -= Ak[i];
		}

		E_oneNorm = oneNorm ( Ek );
		M_oneNorm = oneNorm ( Ak );
		M_infNorm = infNorm ( Ak );
	} while (E_oneNorm > M_oneNorm * eps);

	if (abs(det) < eps){
		for (int i = 0; i < 9; i++){
			if (i == 0 || i == 4 || i == 8)
				Q.val[i] = 1;
			else
				Q.val[i] = 0;
		}
	}
	else {
		// Q = Mk^T 
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				Q.val[3 * i + j] = Ak[3 * j + i];

		/*for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{
		S[3 * i + j] = 0;
		for (int k = 0; k < 3; k++)
		S[3 * i + j] += Ak[3 * i + k] * A[3 * k + j];
		}*/
	}

	buf.mQ[0] = Q;

	//printf ( "mQ = %f %f %f\n", Q.val[0], Q.val[1], Q.val[2] );
	
}

__global__ void computeSolidDeltaPos ( bufList buf, int pnum, float3 mc, float3 restmc )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i > pnum) return;
	if (buf.mtype[i] <= 0) return;

	float3 deltax = make_float3 ( 0, 0, 0 );
	float3 b = buf.mrestpos[i] - restmc;
	float3 Qr = make_float3 ( 0, 0, 0 );
	float9 Q = buf.mQ[0];
	Qr.x = Q.val[0] * b.x + Q.val[1] * b.y + Q.val[2] * b.z;
	Qr.y = Q.val[3] * b.x + Q.val[4] * b.y + Q.val[5] * b.z;
	Qr.z = Q.val[6] * b.x + Q.val[7] * b.y + Q.val[8] * b.z;
	deltax = Qr + mc - buf.mpos[i];

	buf.mpos[i] += deltax;
	//buf.mposbak[i] += deltax;
}

__host__ __device__ mfloat operator/(mfloat a, float b) {
	mfloat c;
	for (int i = 0; i < TYPE_NUM; i++)
		c.tid[i] = a.tid[i] / b;
	return c;
}

__host__ __device__ mfloat operator*(mfloat a, float b) {
	mfloat c;
	for (int i = 0; i < TYPE_NUM; i++)
		c.tid[i] = a.tid[i] * b;
	return c;
}

__host__ __device__ mfloat operator*(float b, mfloat a) {
	mfloat c;
	for (int i = 0; i < TYPE_NUM; i++)
		c.tid[i] = a.tid[i] * b;
	return c;
}

__host__ __device__ mfloat operator+(mfloat a, mfloat b) {
	mfloat c;
	for (int i = 0; i < TYPE_NUM; i++)
		c.tid[i] = a.tid[i] + b.tid[i];
	return c;
}

__host__ __device__ mfloat operator-(mfloat a, mfloat b) {
	mfloat c;
	for (int i = 0; i < TYPE_NUM; i++)
		c.tid[i] = a.tid[i] - b.tid[i];
	return c;
}

__host__ __device__ mfloat3 operator+(mfloat3 a, mfloat3 b) {
	mfloat3 c;
	for (int i = 0; i < TYPE_NUM; i++)
		c.tid[i] = a.tid[i] + b.tid[i];
	return c;
}

__device__ mfloat contributeLaplaceAlpha ( uint i, float3 pos, mfloat ialpha, uint cell, bufList buf )
{
	float3 dist;
	float dsq, c;

	mfloat sum;
	for (int k = 0; k<TYPE_NUM; k++) sum.tid[k] = 0;

	if (buf.mgridcnt[cell] == 0) return sum;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++) {
		uint j = buf.mgrid[cndx];
		if (buf.mtype[j] != 0) continue;
		dist = pos - buf.mpos[j];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (dsq < simData.rd2 && dsq > 0.0) {
			float tt = dsq / (dsq + 0.01*simData.r2);
			dsq = sqrt ( dsq );
			c = (simData.psmoothradius - dsq);
			float vterm = buf.mmass[j] / buf.mrestdens[j];
			//sum = sum + (buf.malpha[j] - ialpha) * simData.lapkern * c;
			sum = sum + (ialpha - buf.malpha[j])*(0.01*0.01*2)*tt*c*c*simData.spikykern / dsq * vterm;
		}
	}

	return sum;
}

__global__ void computeChemicalPotential ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	if (buf.mtype[i] != 0) return;
	for (int k = 0; k < TYPE_NUM; k++) buf.mbeta[i].tid[k] = 0;

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	float3 pos = buf.mpos[i];
	mfloat ialpha = buf.malpha[i];
	float eps = 0.01;
	mfloat lapalpha;
	for (int k = 0; k < TYPE_NUM; k++) lapalpha.tid[k] = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++) {
		lapalpha = lapalpha + contributeLaplaceAlpha ( i, pos, ialpha, gc + simData.gridAdj[c], buf );
	}

	//// F(c0, c1): Helmholtz free energy
	//float F_c0 = 4*ialpha.tid[0]*ialpha.tid[2]*(ialpha.tid[2]-ialpha.tid[0]) + 0.4*ialpha.tid[1] - 0.12;
	//float F_c1 = -4*ialpha.tid[0]*ialpha.tid[0]*ialpha.tid[2]+ 2*(ialpha.tid[0]+0.2)*(ialpha.tid[1]-0.2) - pow((ialpha.tid[1]-0.4), 2) + 2*(ialpha.tid[2]+0.2)*(ialpha.tid[1]-0.4);

	//// chemical potential
	//buf.mbeta[i].tid[0] = F_c0 - 2 * eps*eps*lapalpha.tid[0] - eps*eps*lapalpha.tid[1];
	//buf.mbeta[i].tid[1] = F_c1 - eps*eps*lapalpha.tid[0] - 2 * eps*eps*lapalpha.tid[1];

	//float F_c = (4 * pow ( ialpha.tid[0], 3 ) - 6 * pow ( ialpha.tid[0], 2 ) + 2 * ialpha.tid[0]) / 4.0;
	////F_c = (ialpha.tid[0] - 0.4)*(ialpha.tid[0] - 0.6)*(2 * ialpha.tid[0] - 1) * 4;
	//buf.mbeta[i].tid[0] = F_c - eps*eps*lapalpha.tid[0];

	float gamma = 16.0;
	float F_c[TYPE_NUM];
	F_c[0] = gamma * (ialpha.tid[0]-0.4) * pow ( ialpha.tid[1]-0.4, 2 ) / 2.0;
	F_c[1] = gamma * (ialpha.tid[1]-0.4) * pow ( ialpha.tid[0]-0.4, 2 ) / 2.0;
	/*F_c[0] = gamma * ialpha.tid[0] * pow ( ialpha.tid[1], 2 ) / 2.0;
	F_c[1] = gamma * ialpha.tid[1] * pow ( ialpha.tid[0], 2 ) / 2.0;*/

	float beta = 0; // Lagarangian multiplier
	for (int k = 0; k < TYPE_NUM; k++)
		beta += F_c[k];
	beta /= -1.0*TYPE_NUM;

	for (int k = 0; k < TYPE_NUM; k++)
		buf.mbeta[i].tid[k] = F_c[k] - /*eps*eps**/lapalpha.tid[k] + beta;
}

__device__ mfloat3 contributeFlux ( uint i, float3 pos, mfloat ipoten, uint cell, bufList buf )
{
	float3 dist;
	float dsq, c;

	mfloat3 sum;
	for (int k = 0; k<TYPE_NUM; k++) sum.tid[k] = make_float3 ( 0, 0, 0 );

	if (buf.mgridcnt[cell] == 0) return sum;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++) {
		uint j = buf.mgrid[cndx];
		if (buf.mtype[j] != 0) return;
		dist = pos - buf.mpos[j];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (dsq < simData.rd2 && dsq > 0.0) {
			dsq = sqrt ( dsq * simData.d2 );
			c = (simData.psmoothradius - dsq);
			for (int k = 0; k < TYPE_NUM; k++)
				sum.tid[k] += (buf.mbeta[j].tid[k] - ipoten.tid[k]) * c * c * simData.spikykern / dsq * dist;
		}
	}

	return sum;
}

__global__ void computeFlux ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	if (buf.mtype[i] != 0) return;
	for (int k = 0; k<TYPE_NUM; k++) buf.galpha[i].tid[k] = make_float3(0, 0, 0);

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	float3 pos = buf.mpos[i];
	mfloat ipoten = buf.mbeta[i];
	mfloat3 flux;
	for (int k = 0; k<TYPE_NUM; k++) flux.tid[k] = make_float3 ( 0, 0, 0 );
	for (int c = 0; c < simData.gridAdjCnt; c++) {
		flux = flux + contributeFlux ( i, pos, ipoten, gc + simData.gridAdj[c], buf );
	}

	mfloat Mc;
	for (int k = 0; k < TYPE_NUM; k++)
		Mc.tid[k] = 1.0;

	for (int k = 0; k < TYPE_NUM; k++)
		buf.galpha[i].tid[k] = Mc.tid[k] * flux.tid[k];
}

__device__ mfloat contributeDeltaM ( uint i, float3 pos, mfloat iflux, uint cell, bufList buf )
{
	float3 dist;
	float dsq, c;

	mfloat sum;
	for (int k = 0; k < TYPE_NUM; k++) sum.tid[k] = 0;

	if (buf.mgridcnt[cell] == 0) return sum;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++) {
		uint j = buf.mgrid[cndx];
		if (buf.mtype[j] != 0) continue;
		dist = pos - buf.mpos[j];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (dsq < simData.rd2 && dsq > 0.0) {
			float tt = dsq / (dsq + 0.01*simData.r2);
			dsq = sqrt ( dsq );
			c = (simData.psmoothradius - dsq);
			float vterm = buf.mmass[j] / buf.mrestdens[j];
			/*for (int k = 0; k < TYPE_NUM; k++)
				sum.tid[k] += c * c * simData.spikykern / dsq * dot((buf.galpha[j].tid[k] + iflux.tid[k])/2.0, dist);*/
			sum = sum + (iflux - buf.mbeta[j])*2*tt*c*c*simData.spikykern / dsq * vterm;
		}
	}

	return sum;
}

__global__ void computeDeltaM ( bufList buf, int pnum, float dt )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	if (buf.mtype[i] != 0) return;
	for (int k = 0; k<TYPE_NUM; k++) buf.deltam[i].tid[k] = 0;

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	float3 pos = buf.mpos[i];
	//mfloat3 iflux = buf.galpha[i];
	mfloat iflux = buf.mbeta[i];
	mfloat delta;
	for (int k = 0; k<TYPE_NUM; k++) delta.tid[k] = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++) {
		delta = delta + contributeDeltaM ( i, pos, iflux, gc + simData.gridAdj[c], buf );
	}

	buf.deltam[i] = delta * dt;

	//if (i % 1000 == 0) printf ( "delta = %f %f %f\n", delta.tid[0], delta.tid[1], delta.tid[2] );
	//if (i % 1000 == 0) printf ( "delta = %f %f\n", delta.tid[0], delta.tid[1] );
}

__global__ void updateM ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	if (buf.mtype[i] != 0) return;

	buf.malpha[i] = buf.malpha[i] + buf.deltam[i];
	/*buf.malpha[i].tid[TYPE_NUM - 1] = 1;
	for (int k = 0; k < TYPE_NUM-1; k++)
		buf.malpha[i].tid[TYPE_NUM-1] -= buf.malpha[i].tid[k];*/

	// correction
	for (int k = 0; k < TYPE_NUM; k++) 
		buf.malpha[i].tid[k] = max ( 0.0f, buf.malpha[i].tid[k] );
	float sum = 0;
	for (int k = 0; k < TYPE_NUM; k++)
		sum += buf.malpha[i].tid[k];
	if (sum) {
		for (int k = 0; k < TYPE_NUM; k++)
			buf.malpha[i].tid[k] /= sum;
	}

	mfloat beta;
	
	// update density
	float invrho = 0;
	for (int k = 0; k < TYPE_NUM; k++) {
		invrho += buf.malpha[i].tid[k] / simData.mprest_dens.tid[k];
	}
	buf.mrestdens[i] = 1.0 / invrho;
	
	beta = buf.malpha[i];
	if (TYPE_NUM==2)
		buf.mclr[i] = COLORA ( beta.tid[0], beta.tid[1], 0, 1 );
	else if (TYPE_NUM==3)
		buf.mclr[i] = COLORA ( beta.tid[0], beta.tid[1], beta.tid[2], 1 );

	//if (i % 1000 == 0) printf ( "alpha = %f %f\n", buf.malpha[i].tid[0], buf.malpha[i].tid[1] );
}

__host__ __device__ lambda operator+(lambda a, lambda b) {
	lambda c;
	c.dens = a.dens + b.dens;
	c.sumi = a.sumi + b.sumi;
	c.sumj = a.sumj + b.sumj;

	return c;
}

__device__ lambda contributeLambda ( int i, float3 p, int cell, bufList buf )
{
	float3 dist;
	float dsq, c;
	float s = 1.0;
	//if (buf.mtype[i] != 0) s = 2.0; // ?
	lambda sum;
	sum.dens = 0; sum.sumi = make_float3 ( 0, 0, 0 ); sum.sumj = 0;

	if (buf.mgridcnt[cell] == 0) return sum;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++) {
		uint j = buf.mgrid[cndx];

#ifdef HYBRID
		if (buf.mtype[j] == -3)
			continue;
#endif

		dist = p - buf.mpos[j];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (dsq < simData.r2 && dsq > 0.0) {
			// density
			c = (simData.r2 - dsq);
			if (buf.mtype[j] == 0)
				sum.dens += c * c * c * buf.mmass[i]; // mi or mj
			else if (buf.mtype[j] < 0)
				sum.dens += c * c * c * buf.mmass[i];
			else
				sum.dens += s * c * c * c * buf.mmass[i]; // mi or mj
			// sumi
			dsq = sqrt ( dsq );
			c = (simData.psmoothradius - dsq);
			float3 tmp = buf.mmass[i] * c * c / dsq * dist;
			sum.sumi += tmp;
			// sumj
			float cof = simData.pmass / buf.mmass[j];
			sum.sumj += cof * dot ( tmp, tmp );
		}
	}

	return sum;
}

__global__ void computeLambda ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	buf.mlambda[i] = -0.2;
	//if (buf.mtype[i] < 0) return; // if boundary particles considered ot not
	
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	float3 pos = buf.mpos[i];
	lambda sum;
	sum.dens = 0; sum.sumi = make_float3 ( 0, 0, 0 ); sum.sumj = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++) {
		sum = sum + contributeLambda ( i, pos, gc + simData.gridAdj[c], buf );
	}

	float dens = max ( 1e-6, sum.dens * simData.poly6kern );

	float3 sumi = sum.sumi;
	float sumj = sum.sumj;

	float prest_dens = buf.mrestdens[i];
	float cof = simData.pmass / buf.mmass[i];
	float val = (simData.spikykern*simData.spikykern) * ( cof * dot ( sumi, sumi ) + sumj ) / prest_dens / prest_dens; // ?
	buf.mlambda[i] = -(dens / prest_dens - 1) / (val + simData.lambda_epsi); // ?
	//buf.mlambda[i] = -0.1;
	//if (buf.mtype[i] < 0) buf.mlambda[i] -= 0.2;

	//if (i % 1000 == 0) printf ( "lambda = %f\n", buf.mlambda[i] );
}

__device__ float3 contributeDeltaPos ( uint i, float3 p, float ilambda, uint cell, bufList buf )
{
	float3 dist;
	float dsq, c, corr, k, n, deltaq, cc;

	float3 sum = make_float3 ( 0.0, 0.0, 0.0 );
	corr = 0.0; k = 0.1; n = 1;
	deltaq = 0.4*simData.psmoothradius;
	cc = simData.r2 - deltaq*deltaq;

	if (buf.mgridcnt[cell] == 0) return make_float3 ( 0.0, 0.0, 0.0 );

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++) {
		uint j = buf.mgrid[cndx];

#ifdef HYBRID
		if (buf.mtype[j] == -3)
			continue;
#endif

		dist = p - buf.mpos[j];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (dsq < simData.rd2 && dsq > 0.0) {
			c = (simData.rd2 - dsq);
			corr = -k*pow ( (c*c*c) / (cc*cc*cc), n );
			dsq = sqrt ( dsq );
			c = (simData.psmoothradius - dsq);
			sum += buf.mmass[j] * c * c / dsq * dist * (ilambda + buf.mlambda[j] + corr); // momentum conservation
		}
	}

	return sum;
}

__global__ void computeDeltaPos ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	buf.mdeltapos[i] = make_float3 ( 0.0, 0.0, 0.0 );
	if (buf.mtype[i] < 0) return;

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	register float3 delta = make_float3 ( 0, 0, 0 );

	float3 pos = buf.mpos[i];
	float ilambda = buf.mlambda[i];
	for (int c = 0; c < simData.gridAdjCnt; c++){
		delta += contributeDeltaPos ( i, pos, ilambda, gc + simData.gridAdj[c], buf );
	}
	delta *= simData.spikykern;

	float prest_dens = buf.mrestdens[i];
	float cof = simData.pmass / buf.mmass[i];
	buf.mdeltapos[i] = delta / prest_dens * cof; // ?
	
	//perform collision detection and response

}

__global__ void updatePosition ( float time, float dt, bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	if (buf.mtype[i] < 0) return;

	//// colllision detection and response
	float3 accel = make_float3 ( 0, 0, 0 ), norm, pos = buf.mposbak[i], veval = buf.mvel[i];
	float diff, adj, speed;
	//// Boundaries
	//// Y-axis
	//diff = simData.pradius - (pos.y - (simData.pboundmin.y + (pos.x - simData.pboundmin.x)*simData.pground_slope));
	//if (diff > EPSILON) {
	//	norm = make_float3 ( -simData.pground_slope, 1.0 - simData.pground_slope, 0 );
	//	adj = simData.pextstiff * diff - simData.pdamp * dot ( norm, veval );
	//	norm *= adj; accel += norm;
	//}

	//diff = simData.pradius - (simData.pboundmax.y - pos.y);
	//if (diff > EPSILON) {
	//	norm = make_float3 ( 0, -1, 0 );
	//	adj = simData.pextstiff * diff - simData.pdamp * dot ( norm, veval );
	//	norm *= adj; accel += norm;
	//}

	//// X-axis
	//diff = simData.pradius - (pos.x - (simData.pboundmin.x + (sin ( time*simData.pforce_freq ) + 1)*0.5 * simData.pforce_min));
	//if (diff > EPSILON) {
	//	norm = make_float3 ( 1, 0, 0 );
	//	adj = (simData.pforce_min + 1) * simData.pextstiff * diff - simData.pdamp * dot ( norm, veval );
	//	norm *= adj; accel += norm;
	//}
	//diff = simData.pradius - ((simData.pboundmax.x - (sin ( time*simData.pforce_freq ) + 1)*0.5*simData.pforce_max) - pos.x);
	//if (diff > EPSILON) {
	//	norm = make_float3 ( -1, 0, 0 );
	//	adj = (simData.pforce_max + 1) * simData.pextstiff * diff - simData.pdamp * dot ( norm, veval );
	//	norm *= adj; accel += norm;
	//}

	//// Z-axis
	//diff = simData.pradius - (pos.z - simData.pboundmin.z);
	//if (diff > EPSILON) {
	//	norm = make_float3 ( 0, 0, 1 );
	//	adj = simData.pextstiff * diff - simData.pdamp * dot ( norm, veval );
	//	norm *= adj; accel += norm;
	//}
	//diff = simData.pradius - (simData.pboundmax.z - pos.z);
	//if (diff > EPSILON) {
	//	norm = make_float3 ( 0, 0, -1 );
	//	adj = simData.pextstiff * diff - simData.pdamp * dot ( norm, veval );
	//	norm *= adj; accel += norm;
	//}

	// Accel Limit
	/*speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
	if (speed > simData.AL2) {
		accel *= simData.AL / sqrt ( speed );
	}*/

	// Velocity Limit
	/*float3 vel = buf.mvel[i];
	float speed = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
	if (speed > simData.VL2) {
		speed = simData.VL2;
		vel *= simData.VL / sqrt ( speed );
	}*/

	float R = 40;
	float2 pp = make_float2 ( buf.mpos[i].x, buf.mpos[i].z );
	/*float len = length ( pp );
	diff = len - R;
	if (diff > EPSILON) {
		norm = make_float3 ( -pp.x/len, 0, -pp.y/len );
		adj = simData.pextstiff * diff - simData.pdamp * dot ( norm, veval );
		norm *= adj; accel += norm;
	}
	float3 voffset = accel * dt;*/
	buf.mpos[i] += buf.mdeltapos[i] /*+ voffset * dt*/;

	//printf("ID: %d DeltaPos: (%f,%f,%f)\n", i, buf.mdeltapos[i].x, buf.mdeltapos[i].y, buf.mdeltapos[i].z);
	//float3 dx = make_float3 ( simData.pradius, simData.pradius, simData.pradius );
	//buf.mpos[i] = clamp ( buf.mpos[i], simData.pboundmin + dx, simData.pboundmax - dx );
#ifndef HYBRID
	if (length ( pp ) > R) {
		buf.mpos[i].x = R*pp.x / length ( pp );
		buf.mpos[i].z = R*pp.y / length ( pp );
		/*buf.mtype[i] = -1;
		buf.mclr[i] = COLORA ( 1, 1, 1, 1 );*/
	}
	
	buf.mpos[i].y = max ( 14.0, buf.mpos[i].y );
#endif
	//if (buf.mpos[i].y <= 9.0) buf.mclr[i] = COLORA ( 1, 1, 1, 0 );
}

__global__ void updateVelocity ( bufList buf, int pnum, float dt )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	if (buf.mtype[i] < 0) return;

	buf.mvel[i] = (buf.mpos[i] - buf.mposbak[i]) / dt;   // ?
	//printf("ID: %d, Vel: (%f,%f,%f)\n", i, buf.mvel[i].x, buf.mvel[i].y, buf.mvel[i].z);
}

__device__ float3 contributeVort ( uint i, float3 p, float3 vel, uint cell, bufList buf )
{
	float3 dist;
	float dsq, c;

	float3 sum = make_float3 ( 0.0, 0.0, 0.0 );
	if (buf.mgridcnt[cell] == 0) return make_float3 ( 0.0, 0.0, 0.0 );

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++) {
		uint j = buf.mgrid[cndx];
		if (buf.mtype[j] != 0) continue;
		dist = p - buf.mpos[j];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (dsq < simData.rd2 && dsq > 0.0) {
			dsq = sqrt ( dsq );
			c = (simData.psmoothradius - dsq);
			float3 tmp = cross ( buf.mvel[j] - vel, dist );
			sum += c * c * simData.spikykern / dsq * tmp;
		}
	}
	return sum;
}

__global__ void computeVorticity ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	buf.mw[i] = make_float3 ( 0.0, 0.0, 0.0 );
	if (buf.mtype[i] != 0) return;

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	register float3 tmpw = make_float3 ( 0.0, 0.0, 0.0 );
	float3 pos = buf.mpos[i];
	float3 vel = buf.mvel[i];

	for (int c = 0; c < simData.gridAdjCnt; c++){
		tmpw += contributeVort ( i, pos, vel, gc + simData.gridAdj[c], buf );
	}
	
	buf.mw[i] = tmpw;
}

__device__ float3 contributeN ( uint i, float3 p, uint cell, bufList buf )
{
	float3 dist;
	float dsq, c, w;

	float3 sum = make_float3 ( 0.0, 0.0, 0.0 );
	if (buf.mgridcnt[cell] == 0) return make_float3 ( 0.0, 0.0, 0.0 );

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++) {
		uint j = buf.mgrid[cndx];
		if (buf.mtype[j] != 0) continue;
		dist = p - buf.mpos[j];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (dsq < simData.rd2 && dsq > 0.0) {
			dsq = sqrt ( dsq );
			c = (simData.psmoothradius - dsq);
			w = length ( buf.mw[j] );
			sum += c * c * simData.spikykern / dsq * dist * w;
		}
	}
	return sum;
}

__global__ void computeNormal ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	buf.mN[i] = make_float3 ( 0.0, 0.0, 0.0 );
	if (buf.mtype[i] != 0) return;

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	register float3 tmpN = make_float3 ( 0.0, 0.0, 0.0 );
	float3 pos = buf.mpos[i];
	for (int c = 0; c < simData.gridAdjCnt; c++){
		tmpN += contributeN ( i, pos, gc + simData.gridAdj[c], buf );
	}
	float len = length ( tmpN );
	if (len){
		tmpN /= len;
	}
	
	buf.mN[i] = tmpN;
}

__global__ void vorticityConfinement ( bufList buf, int pnum, float dt )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	if (buf.mtype[i] != 0) return;

	buf.mvel[i] += dt/*/buf.mmass[i]*/ * simData.vort_epsi * cross ( buf.mN[i], buf.mw[i] );  // ?
}

__device__ float3 contributeXSPH ( uint i, float3 p, float3 vel, mfloat ialpha, uint cell, bufList buf )
{
	float3 dist;
	float dsq, c;

	float3 sum = make_float3 ( 0.0, 0.0, 0.0 );
	if (buf.mgridcnt[cell] == 0) return make_float3 ( 0.0, 0.0, 0.0 );

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++) {
		uint j = buf.mgrid[cndx];
		if (buf.mtype[j] != 0) continue; // boundary particles
		dist = p - buf.mpos[j];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (dsq < simData.rd2 && dsq > 0.0) {
			c = (simData.rd2 - dsq);
			sum += c * c * c * (buf.mvel[j] - vel);

			// gradient of c
			/*dsq = sqrt ( dsq * simData.d2 );
			c = (simData.psmoothradius - dsq);
			for (int k = 0; k < TYPE_NUM; k++)
				buf.galpha[i].tid[k] += (buf.malpha[j].tid[k] - ialpha.tid[k]) * c * c * simData.spikykern / dsq * dist;*/
		}
	}

	return sum;
}

__global__ void computeXSPHViscosity ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	buf.mveleval[i] = make_float3 ( 0.0, 0.0, 0.0 );
	for (int k = 0; k < TYPE_NUM; k++) buf.galpha[i].tid[k] = make_float3 ( 0, 0, 0 );
	if (buf.mtype[i] != 0) return;

	register float3 tmpv = make_float3 ( 0.0, 0.0, 0.0 );
	float cc = 0.5;

	//if (buf.mmass[i] >= simData.pmass*0.9) cc = 1;

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	float3 pos = buf.mpos[i];
	float3 vel = buf.mvel[i];
	mfloat ialpha = buf.malpha[i];
	for (int c = 0; c < simData.gridAdjCnt; c++){
		tmpv += contributeXSPH ( i, pos, vel, ialpha, gc + simData.gridAdj[c], buf );
	}
#ifndef HYBRIDSPHSIM
	buf.mveleval[i] = cc * tmpv * simData.poly6kern;
#else
	buf.mrestpos[i] = cc * tmpv * simData.poly6kern; //XSPH
#endif
}

__global__ void updateXSPHVelocity ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	if (buf.mtype[i] != 0) return;
#ifndef HYBRIDSPHSIM
	buf.mvel[i] += buf.mveleval[i];
#endif
}

__device__ mfloat3 contributeGradientC ( uint i, float3 pos, mfloat ialpha, uint cell, bufList buf )
{
	float3 dist;
	float dsq, c;

	mfloat3 sum;
	for (int k = 0; k<TYPE_NUM; k++) sum.tid[k] = make_float3 ( 0, 0, 0 );

	if (buf.mgridcnt[cell] == 0) return sum;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++) {
		uint j = buf.mgrid[cndx];
		if (buf.mtype[j] != 0) return;
		dist = pos - buf.mpos[j];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (dsq < simData.rd2 && dsq > 0.0) {
			dsq = sqrt ( dsq * simData.d2 );
			c = (simData.psmoothradius - dsq);
			for (int k = 0; k < TYPE_NUM; k++)
				sum.tid[k] += (buf.malpha[j].tid[k] - ialpha.tid[k]) * c * c * simData.spikykern / dsq * dist;
		}
	}

	return sum;
}

__global__ void computeGradientC ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	if (buf.mtype[i] != 0) return;
	for (int k = 0; k < TYPE_NUM; k++) buf.galpha[i].tid[k] = make_float3 ( 0, 0, 0 );

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	float3 pos = buf.mpos[i];
	mfloat3 gradc;
	mfloat ialpha = buf.malpha[i];
	for (int k = 0; k < TYPE_NUM; k++) gradc.tid[k] = make_float3 ( 0, 0, 0 );
	for (int c = 0; c < simData.gridAdjCnt; c++){
		gradc = gradc + contributeGradientC ( i, pos, ialpha, gc + simData.gridAdj[c], buf );
	}

	buf.galpha[i] = gradc;
}

__device__ mfloat contributeSF ( uint i, float3 pos, mfloat3 igrad, uint cell, bufList buf )
{
	float3 dist;
	float dsq, c;

	mfloat sum;
	for (int k = 0; k < TYPE_NUM; k++) sum.tid[k] = 0;

	if (buf.mgridcnt[cell] == 0) return sum;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++) {
		uint j = buf.mgrid[cndx];
		if (buf.mtype[j] != 0) continue;
		dist = pos - buf.mpos[j];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (dsq < simData.rd2 && dsq > 0.0) {
			dsq = sqrt ( dsq * simData.d2 );
			c = (simData.psmoothradius - dsq);
			for (int k = 0; k < TYPE_NUM; k++) {
				float len = length ( buf.galpha[j].tid[k] );
				if (!len) len = 1.0;
				sum.tid[k] += c * c * simData.spikykern / dsq * dot ( (buf.galpha[j].tid[k] / len + igrad.tid[k]) / 2.0, dist );
			}
		}
	}

	return sum;
}

__global__ void computeSF ( bufList buf, int pnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	if (buf.mtype[i] != 0) return;

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	float3 pos = buf.mpos[i];
	mfloat sf;
	mfloat3 igrad = buf.galpha[i];
	for (int k = 0; k < TYPE_NUM; k++){
		sf.tid[k] = 0;
		float len = length ( igrad.tid[k] );
		if (len)
			igrad.tid[k] /= len;
	}
	for (int c = 0; c < simData.gridAdjCnt; c++){
		sf = sf + contributeSF ( i, pos, igrad, gc + simData.gridAdj[c], buf );
	}

	float3 deltav = make_float3 ( 0, 0, 0 );
	float eps = 0.01;
	float sigma[TYPE_NUM][TYPE_NUM];
	sigma[0][1] = 1.0;
	for (int ii = 0; ii < TYPE_NUM - 1; ii++)
		for (int jj = ii + 1; jj < TYPE_NUM; jj++) {
			float3 sum = sf.tid[ii] * length ( buf.galpha[i].tid[ii]) * buf.galpha[i].tid[ii] + sf.tid[jj] * length ( buf.galpha[i].tid[jj]) * buf.galpha[i].tid[jj];
			deltav += sigma[ii][jj] / 2.0*(-6 * sqrt ( 2.0 )*eps*sum) * 5 * buf.malpha[i].tid[ii] * buf.malpha[i].tid[jj];
		}
	buf.mvel[i] += deltav / buf.mrestdens[i];
}

__global__ void updateParticlePosition ( bufList buf, int pnum, int frame )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;
	if (i >= pnum) return;
	if (buf.mtype[i] != -2) return;

	float3 pos = buf.mpos[i];
	float theta;

	if (frame > 250 && frame < 400)
		buf.mpos[i].y -= 0.72;
	theta = 1.0*M_PI / 150.0;
	if (frame>400 && frame <= 700)
		buf.mpos[i] = make_float3 ( cos ( theta )*pos.x - sin ( theta )*pos.z, pos.y, sin ( theta )*pos.x + cos ( theta )*pos.z );

	if (frame>800 && frame<=1100)
		buf.mpos[i] = make_float3 ( cos ( theta )*pos.x + sin ( theta )*pos.z, pos.y, -sin ( theta )*pos.x + cos ( theta )*pos.z );

	/*if (frame>1200 && frame <= 6000)
		buf.mpos[i] = make_float3 ( cos ( theta )*pos.x - sin ( theta )*pos.z, pos.y, sin ( theta )*pos.x + cos ( theta )*pos.z );*/

	/*if (frame==1000)
		buf.mpos[i].y = -1000;*/
	/*if (frame < 200) {
		theta = 1.0 / 200.0 * M_PI;
		pos.y -= thre;
		buf.mpos[i] = make_float3 ( cos ( theta )*pos.x - sin ( theta )*pos.y, sin ( theta )*pos.x + cos ( theta )*pos.y + thre, pos.z );
	}

	if (frame > 400 && frame < 1200) {
		theta = 1.0 / 400.0 * M_PI;
		if (frame>=500 && (frame-500)/200%2==0)
			theta *= -1;

		pos.y -= thre;
		buf.mpos[i] = make_float3 ( cos ( theta )*pos.x - sin ( theta )*pos.y, sin ( theta )*pos.x + cos ( theta )*pos.y + thre, pos.z );
	}

	if (frame>1300 && frame < 1500) {
		theta = 1.0 / 200.0 * M_PI;
		pos.y -= thre;
		buf.mpos[i] = make_float3 ( cos ( theta )*pos.x - sin ( theta )*pos.y, sin ( theta )*pos.x + cos ( theta )*pos.y + thre, pos.z );
	}*/

	/*if (frame>2600 && frame < 3400) {
		theta = 1.0 / 800.0 * M_PI;
		if (frame >= 2700 && (frame - 2700) / 200 % 2 == 0)
			theta *= -1;
		pos.y -= thre;
		buf.mpos[i] = make_float3 ( cos ( theta )*pos.x - sin ( theta )*pos.y, sin ( theta )*pos.x + cos ( theta )*pos.y + thre, pos.z );
	}*/
}


void updateSimParams ( FluidParams* cpufp )
{
	cudaError_t status;
	#ifdef CUDA_42
		// Only for CUDA 4.x or earlier. Depricated in CUDA 5.0+
		// Original worked even if symbol was declared __device__
		status = cudaMemcpyToSymbol ( "simData", cpufp, sizeof(FluidParams) );
	#else
		// CUDA 5.x+. Only works if symbol is declared __constant__
		status = cudaMemcpyToSymbol ( simData, cpufp, sizeof(FluidParams) );
	#endif

	/*app_printf ( "SIM PARAMETERS:\n" );
	app_printf ( "  CPU: %p\n", cpufp );	
	app_printf ( "  GPU: %p\n", &simData );	 */
}

#ifdef HYBRID
__device__ float labelNearestSWEvertex(int i, float3 pos, int cell, bufList buf, float tempmax)
{
	float3 dist;
	float dsq, c, sum;
	float massj;
	//register float d2 = simData.psimscale * simData.psimscale;
	//register float r2 = simData.r2 / d2;

	float searchR2 = LABEL_DIS * LABEL_DIS;
	float inmax = tempmax;
	//int j, isboundj, isboundi;

	//register float cmterm;
	////register float3 alphagrad[MAX_FLUIDNUM];

	//sum = 0.0;
	//float tempkern = -10.0;

	if (buf.mgridcnt[cell] == 0) return tempmax;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	for (int cndx = cfirst; cndx < clast; cndx++) {
		uint j = buf.mgrid[cndx];
		if (buf.mtype[j] == -3 && buf.mpos[j].y>-15.0f)  ///this >-15.0f thing is just for case testing
		{
			dist = pos - buf.mpos[j];
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

			if (dsq <= searchR2 && dsq<inmax)
			{
				inmax = dsq;
				buf.sweVindex[i] = buf.sweVindex[j];
				//printf("labeled\n");
			}
		}

	}

	return inmax;
}


__global__ void prepareLabelParticles(bufList buf, int pnum)
{

	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;

	if (buf.mtype[i] < 0) return;

	buf.sweVindex[i] = -1;
	float tempmax = 88888.0f;

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	//float tempkern = -10.0, kbuf;

	float3 pos = buf.mpos[i];


	for (int c = 0; c<simData.gridAdjCnt; c++)
	{
		tempmax = labelNearestSWEvertex(i, pos, gc + simData.gridAdj[c], buf, tempmax);  //grid location index are the same for first & second simulation grids
	}

	if (tempmax>LABEL_DIS*LABEL_DIS) //no need?
		buf.sweVindex[i] = -1;

}
#endif