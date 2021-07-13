#pragma once

#include "../../../math/geometry.h"

namespace msph {

class MultiphaseParam
{
public:
    int numFluidParticles;
    int numSolidParticles;

    float   dx;
    cfloat3 gridxmin;
    cfloat3 gridxmax;
    cint3   gridres;

    cfloat3 gravity;
    float   dt;
    float   viscosity;
    float   restdensity;
    float   pressureK;
    float   acceleration_limit;

    float spacing;
    float smoothradius;
    float vol0;

    float klaplacian;
    float kernel_cubic;
    float kernel_cubic_gradient;

    float boundary_visc;
    float boundary_friction;

    int   numTypes;
    float densArr[10];
    float viscArr[10];
    float drift_dynamic_diffusion;
    float drift_turbulent_diffusion;
    float drift_thermal_diffusion;
    float surface_tension;

    float E;               //Young's modulus
    float hourglass_corr;  //a negative value!
    float solidK;
    float solidG;
    float Yield;
    float solid_visc;
    float dragForce;
    float plastic_flow;
    float granularFriction;
    float cohesion;  //cohesion

    int   dissolution;
    float max_alpha[10];  //max volume fraction of each single phase

    float heat_flow_rate;
    float melt_point;
    float latent_heat;
    float heat_capacity[10];

    bool enable_dfsph;
    bool enable_solid;
    bool enable_diffusion;
    bool enable_heat;
};

template <int N>
struct Volfrac
{
    float data[N];
};
using volf = Volfrac<2>;

struct SimDataMultiphase
{
    cfloat3* pos;
    cfloat4* color;
    cfloat3* vel;
    int*     type;
    int*     uniqueId;
    float*   mass;
    float*   density;
    float*   pressure;
    float*   restDensity;
    float*   viscosity;
    cfloat3* normal;
    cfloat3* force;

    int* indexTable;

    uint* particleHash;
    uint* particleIndex;
    uint* gridCellStart;
    uint* gridCellEnd;
    uint* neighborList;

    // Multiphase
    int*     group;
    volf*    vFrac;
    volf*    vfrac_change;
    float*   effective_mass;
    cfloat3* drift_accel;
    cfloat3* drift_v;
    cfloat3* vol_frac_gradient;
    cfloat3* M_m;
    cmat3*   umkumk;

    //heat phase change
    float* temperature;
    float* heat_buffer;
    float* sorted_temperature;
    float* sorted_heat_buffer;

    //Incompressible SPH
    float* error;
    //IISPH
    cfloat3* dii;
    cfloat3* sum_dp;
    float*   aii;
    float*   new_p;
    float*   density_adv;
    cfloat3* vel_adv;

    //DFSPH
    float* DF_factor;
    float* pstiff;
    float* rho_stiff;
    float* div_stiff;

    //solid
    int*   local_id;
    cmat3* stress;
    cmat3* stressMix;
    cmat3* L;  //kernel correction
};

}  // namespace msph