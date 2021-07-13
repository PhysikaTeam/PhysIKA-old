#pragma once
#include "../math/geometry.h"
#include "ParticleSolver.h"

typedef float scalar;

struct Grid
{
    cint3 dim;  //effective resolution
    int   padding;
    int   xlen, ylen, zlen;  //cell center dimension
    int   ulen, vlen, wlen;  //face dimension

    int Size;                 //full cell number
    int dimSize;              //effective cell number
    int uSize, vSize, wSize;  //full face number

    cfloat3 xmin;  //coordinates of the low corner
    scalar  h;     //cell size

    void setSize()
    {
        xlen = dim.x + padding * 2;
        ylen = dim.y + padding * 2;
        zlen = dim.z + padding * 2;
        ulen = xlen + 1;
        vlen = ylen + 1;
        wlen = zlen + 1;

        Size    = xlen * ylen * zlen;
        uSize   = ulen * ylen * zlen;
        vSize   = xlen * vlen * zlen;
        wSize   = xlen * ylen * wlen;
        dimSize = dim.x * dim.y * dim.z;
    }
    //effective cell id
    int cellId(int x, int y, int z)
    {
        return (z * dim.y + y) * dim.x + x;
    }
    int uId(int x, int y, int z)
    {
        return ((z + padding) * ylen + y + padding) * ulen + x + padding;
    }
    int vId(int x, int y, int z)
    {
        return ((z + padding) * vlen + y + padding) * xlen + x + padding;
    }
    int wId(int x, int y, int z)
    {
        return ((z + padding) * ylen + y + padding) * xlen + x + padding;
    }
};

class GridSolver : public ParticleSolver
{
public:
    Grid grid;

    vecf3 pos;
    vecf4 color;
    vecf3 vel;

    //face-centered
    scalar* u;  //velocity u
    scalar* v;  //velocity v
    scalar* w;  //velocity w
    scalar* uadv;
    scalar* vadv;
    scalar* wadv;

    //cell-centered
    scalar* p;     //pressure
    scalar* divU;  //velocity divergence
    scalar* b;     // tmp: right hand side
    scalar* Aq;    // tmp: A * conjugate basis
    scalar* r;     // tmp: residual
    scalar* q;     // tmp: conjugate basis 0

    //param
    scalar dt;
    scalar rho;
    scalar divUsum;
    int    frame;
    int    pad;

    // ==============   METHOD   =================

    void Step();
    void HandleKeyEvent(char key);

    void   setup();
    void   loadConfig();
    void   allocate();
    void   mvproduct(scalar* v, scalar* dst);
    scalar dotproduct(scalar* v1, scalar* v2);

    cint3   locateCell(cfloat3 p);
    scalar  sampleU(cfloat3 p);  //interpolate velocity
    scalar  sampleV(cfloat3 p);
    scalar  sampleW(cfloat3 p);
    cfloat3 sampleVelocity(cfloat3 p);
    scalar  sampleQ(cfloat3 p, scalar* v);  //interpolate cell-centered value

    void testcase();
    void divVelocity();
    void makeRHS();
    void solve();
    void updateU();
    void bodyForce();
    void advect();
    void swapBuffer();

    void addParticles();
    void advectParticles();
};