#pragma once
#ifndef _SANDGRID_H
#define _SANDGRID_H


#include "types.h"
#include "Core/DataTypes.h"
#include <vector>

#ifndef __CUDACC__  
#define __CUDACC__  
#include "cuda_texture_types.h"  
#endif 

#include "Core/Array/ArrayPitch2D.h"
#include "Core/Array/Array.h"

#include "Dynamics/HeightField/HeightFieldGrid.h"

//using namespace PhysIKA;

namespace PhysIKA
{
    /**
    *@brief Sand height field information
    *@detail Sand info is saved in a (1+nx+1)*(1+ny+1) grid. And the boundary data is just a copy of its neighbor.
    *@detail For each grid, (h, hu, hv, b) data is saved in a float4 struct.
    */
    struct SandGridInfo
    {
        int nx;
        int ny;
        int pitch;

        float4* data=0;

        float sandRho;
        float mu;
        float slide;
        float drag;

        float griddl;
    };


    /**
    *@brief Position information of a sub-block in a grid.
    */
    struct SubBlock
    {
        int gxmin = 0;
        int gymin = 0;

        int gxsize = 0;
        int gysize = 0;
    };



    class SandGrid
    {
    public:
        SandGrid() {}
        SandGrid(int in_x, int in_y, float slide1, float mu1, float drag1);
        ~SandGrid();

        void initialize(float *land_height, float *sand_height);

        //void vis(float4* watersurfacevertices, rgb* watersurfacecolors);

        //void vis(DeviceArray<Vector3f>& visualCoord);

        void getSandGridInfo(SandGridInfo& info);

        DeviceArrayPitch2D4f& getSandGridData() { return m_sandGrid; }
        const DeviceArrayPitch2D4f& getSandGridData() const { return m_sandGrid; }

        void setSandInfo(const SandGridInfo& sandinfo);

        bool resize(int nx, int ny);
        void getSize(int& nx, int& ny)const { nx = Nx; ny = Ny; }

        void setGridLength(float gridLength) { dl = gridLength; }
        float getGridLength()const { return dl; }

        void setMu(float mu_) { mu = mu_; }
        float getMu() const { return mu; }

        void setSlide(float slide_) { slide = slide_; }
        float getSlide()const { return slide; }

        void setDrag(float drag_) { drag = drag_; }
        float getDrag()const { return drag; }

        void setDensity(float rho_) { m_sandRho = rho_; }
        float getDensity()const { return m_sandRho; }

        void updateSandGridHeight();
        void updateLandGridHeight();

        void setSandGridHeight(DeviceHeightField1d& sandheight);
        void setSandGridHeight(HostHeightField1d& sandheight);
        void setSandGridHeight(double* sandheight);
        void getSandGridHeight(double* sandheight);



    private:
        //void grid_init();


    public:
        //int in_Nx;            // actual simulation grid
        //int in_Ny;
        int Nx=0 ;                // data grid size. Use extra 2 columns and 2 rows to save boudary data.
        int Ny=0;

        int block_x = 16;        // block size of GPU thread
        int block_y = 16;        // block size of GPU thread

        float slide;
        float drag;
        float mu;
        float m_sandRho = 5.0;

        float dl = 1.0;

        DeviceArrayPitch2D4f m_sandGrid;

        //DeviceArrayPitch2D1f m_landHeight;
        //DeviceArrayPitch2D1f m_sandHeight;
        DeviceHeightField1d m_sandHeight;
        DeviceHeightField1d m_landHeight;

    };




}



#endif