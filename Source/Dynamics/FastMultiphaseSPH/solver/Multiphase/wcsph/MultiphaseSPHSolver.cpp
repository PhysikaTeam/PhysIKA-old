
#include "cuda_runtime.h"
#include "../../../math/cpTimer.h"
#include "../../../math/math.h"

#include "MultiphaseSPHSolver.h"
#include "multiphase_WCSPH.cuh"

//#define BUILD_NEIGHBOR_LIST

namespace msph {

extern MultiphaseParam* pParamStatic;

void MultiphaseSPHSolver::preinit()
{
    setOutputDir(std::string("../data/"));

    SetupFluidScene();
}
void MultiphaseSPHSolver::postinit()
{
    cudaMallocManaged(&pParam, sizeof(MultiphaseParam));
    memcpy(pParam, &h_param, sizeof(MultiphaseParam));
    pParamStatic = pParam;

    setupDeviceData();
    sortParticles(sortFlagsWithReorder);
    updateMass_host(num_particles);
    computeRigidVolume_host(num_particles);
}

void LoadBox(ParticleObject* particleObject,
             cfloat3         xmin,
             cfloat3         xmax,
             float           spacing)
{
    if (!particleObject)
    {
        printf("Error: trying to load particles into invalid particle object.\n");
        return;
    }
    int id = 0;
    //x-y
    for (float x = xmin.x; x <= xmax.x + EPSILON; x += spacing)
        for (float y = xmin.y; y <= xmax.y + EPSILON; y += spacing)
        {
            particleObject->pos.push_back(cfloat3(x, y, xmin.z));
            particleObject->normal.push_back(cfloat3(0, 0, 1));
            particleObject->type.push_back(TYPE_RIGID);
            particleObject->id.push_back(id);

            particleObject->pos.push_back(cfloat3(x, y, xmax.z));
            particleObject->normal.push_back(cfloat3(0, 0, -1));
            particleObject->type.push_back(TYPE_RIGID);
            particleObject->id.push_back(id);
        }
    //y-z
    for (float z = xmin.z; z <= xmax.z + EPSILON; z += spacing)
        for (float y = xmin.y; y <= xmax.y + EPSILON; y += spacing)
        {
            particleObject->pos.push_back(cfloat3(xmin.x, y, z));
            particleObject->normal.push_back(cfloat3(1, 0, 0));
            particleObject->type.push_back(TYPE_RIGID);
            particleObject->id.push_back(id);

            particleObject->pos.push_back(cfloat3(xmax.x, y, z));
            particleObject->normal.push_back(cfloat3(-1, 0, 0));
            particleObject->type.push_back(TYPE_RIGID);
            particleObject->id.push_back(id);
        }
    //x-z
    for (float x = xmin.x; x <= xmax.x + EPSILON; x += spacing)
        for (float z = xmin.z; z <= xmax.z + EPSILON; z += spacing)
        {
            particleObject->pos.push_back(cfloat3(x, xmin.y, z));
            particleObject->normal.push_back(cfloat3(0, 1, 0));
            particleObject->type.push_back(TYPE_RIGID);
            particleObject->id.push_back(id);

            particleObject->pos.push_back(cfloat3(x, xmax.y, z));
            particleObject->normal.push_back(cfloat3(0, -1, 0));
            particleObject->type.push_back(TYPE_RIGID);
            particleObject->id.push_back(id);
        }
}

void MultiphaseSPHSolver::setupDeviceData()
{
    allocateDeviceBuffer();
    copyData2Device(0, num_particles);
    CopyParamToDevice(h_param);

    copyDataPtrToDevice(simdata);
}

void MultiphaseSPHSolver::allocateDeviceBuffer()
{
    num_particles = pos.size();
    auto buffersz = max_nump;
    cudaMalloc(&simdata.pos, buffersz * sizeof(cfloat3));
    cudaMalloc(&simdata.color, buffersz * sizeof(cfloat4));
    cudaMalloc(&simdata.vel, buffersz * sizeof(cfloat3));
    cudaMalloc(&simdata.type, buffersz * sizeof(int));
    cudaMalloc(&simdata.group, buffersz * sizeof(int));
    cudaMalloc(&simdata.uniqueId, buffersz * sizeof(int));
    cudaMalloc(&simdata.mass, buffersz * sizeof(float));
    cudaMalloc(&simdata.restDensity, buffersz * sizeof(float));
    cudaMalloc(&simdata.normal, buffersz * sizeof(cfloat3));
    cudaMalloc(&simdata.viscosity, buffersz * sizeof(float));
    cudaMalloc(&simdata.density, buffersz * sizeof(float));
    cudaMalloc(&simdata.pressure, buffersz * sizeof(float));

    bufman.alloc_device_only(simdata.force, buffersz);

    bufman.alloc_sortable(simdata.pos, buffersz);
    bufman.alloc_sortable(simdata.color, buffersz);
    bufman.alloc_sortable(simdata.vel, buffersz);
    bufman.alloc_sortable(simdata.type, buffersz);
    bufman.alloc_sortable(simdata.group, buffersz);
    bufman.alloc_sortable(simdata.uniqueId, buffersz);
    bufman.alloc_sortable(simdata.mass, buffersz);
    bufman.alloc_sortable(simdata.restDensity, buffersz);
    bufman.alloc_sortable(simdata.normal, buffersz);
    bufman.alloc_sortable(simdata.viscosity, buffersz);
    bufman.alloc_sortable(simdata.density, buffersz);

    num_cells = h_param.gridres.prod();
    cudaMalloc(&simdata.particleHash, buffersz * sizeof(uint));
    cudaMalloc(&simdata.particleIndex, buffersz * sizeof(uint));
    cudaMalloc(&simdata.gridCellStart, num_cells * sizeof(uint));
    cudaMalloc(&simdata.gridCellEnd, num_cells * sizeof(uint));
#ifdef BUILD_NEIGHBOR_LIST
    cudaMalloc(&simdata.neighborList, buffersz * sizeof(uint) * NUM_NEIGHBOR);
#endif

    //multiphase buffer
    int num_mp = buffersz * h_param.numTypes;
    cudaMalloc(&simdata.vFrac, buffersz * sizeof(msph::volf));
    cudaMalloc(&simdata.vfrac_change, buffersz * sizeof(msph::volf));
    cudaMalloc(&simdata.drift_v, buffersz * sizeof(cfloat3) * h_param.numTypes);
    cudaMalloc(&simdata.drift_accel, buffersz * sizeof(cfloat3));
    cudaMalloc(&simdata.M_m, buffersz * sizeof(cfloat3));
    cudaMalloc(&simdata.umkumk, buffersz * sizeof(cmat3));
    bufman.alloc_sortable(simdata.vFrac, buffersz);
    bufman.alloc_sortable(simdata.M_m, buffersz);
    bufman.alloc_sortable(simdata.umkumk, buffersz);

    bufman.allocHostDeviceBuffer(stress, simdata.stress, buffersz);
    bufman.alloc_sortable(simdata.stress, buffersz);
    bufman.alloc_device_only(simdata.L, buffersz);
    bufman.alloc_device_only(simdata.stressMix, buffersz);

    bufman.setupDeviceSortHandlers();
}

void MultiphaseSPHSolver::copyData2Device(int start, int end)
{
    auto sz = end - start;

    cudaMemcpy(simdata.pos + start, pos.data() + start, sz * sizeof(cfloat3), cudaMemcpyHostToDevice);
    cudaMemcpy(simdata.color + start, color.data() + start, sz * sizeof(cfloat4), cudaMemcpyHostToDevice);
    cudaMemcpy(simdata.vel + start, vel.data() + start, sz * sizeof(cfloat3), cudaMemcpyHostToDevice);
    cudaMemcpy(simdata.normal + start, normal.data() + start, sz * sizeof(cfloat3), cudaMemcpyHostToDevice);

    cudaMemcpy(simdata.type + start, type.data() + start, sz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(simdata.uniqueId + start, unique_id.data() + start, sz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(simdata.mass + start, mass.data() + start, sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(simdata.restDensity + start, rest_density.data() + start, sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(simdata.density + start, rest_density.data() + start, sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(simdata.group + start, group.data() + start, sz * sizeof(int), cudaMemcpyHostToDevice);

    auto num_particlesT = sz * h_param.numTypes;
    auto start_n        = start * h_param.numTypes;
    cudaMemcpy(simdata.vFrac + start, vol_frac.data() + start_n, num_particlesT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(simdata.stress + start, stress.data() + start, sizeof(cmat3) * sz, cudaMemcpyHostToDevice);
}

void MultiphaseSPHSolver::copySimulationDataFromDevice()
{
    //to be worked
    cudaMemcpy(pos.data(), simdata.pos, num_particles * sizeof(cfloat3), cudaMemcpyDeviceToHost);
    //cudaMemcpy(color.data(), simdata.color, num_particles * sizeof(cfloat4), cudaMemcpyDeviceToHost);
    cudaMemcpy(vel.data(), simdata.vel, num_particles * sizeof(cfloat3), cudaMemcpyDeviceToHost);
    //cudaMemcpy(normal.data(), simdata.normal, num_particles * sizeof(cfloat3), cudaMemcpyDeviceToHost);
    //cudaMemcpy(drift_accel.data(), simdata.drift_accel, num_particles * sizeof(cfloat3), cudaMemcpyDeviceToHost);

    cudaMemcpy(type.data(), simdata.type, num_particles * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(unique_id.data(), simdata.uniqueId, num_particles * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(group.data(), simdata.group, num_particles * sizeof(int), cudaMemcpyDeviceToHost);

    auto num_particlesT = num_particles * h_param.numTypes;
    cudaMemcpy(vol_frac.data(), simdata.vFrac, num_particlesT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(stress.data(), simdata.stress, num_particles * sizeof(cmat3), cudaMemcpyDeviceToHost);
}

void MultiphaseSPHSolver::copyPosColorFromDevice()
{
    math::cTime cp;
    cp.tick();
    cudaMemcpy(pos.data(), simdata.pos, num_particles * sizeof(cfloat3), cudaMemcpyDeviceToHost);
    cudaMemcpy(color.data(), simdata.color, num_particles * sizeof(cfloat4), cudaMemcpyDeviceToHost);
    //printf("pos color copied from device, taking %f ms\n", cp.tack()*1000);
}
void MultiphaseSPHSolver::prepareRenderData(cfloat3* dptr_pos, cfloat4* dptr_clr)
{
    cudaMemcpy(dptr_pos, simdata.pos, num_particles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dptr_clr, simdata.color, num_particles * sizeof(cfloat4), cudaMemcpyDeviceToDevice);
}

//============================================
//
// Scene Configuration
//
//============================================

void MultiphaseSPHSolver::SetupFluidScene()
{
    SetParameter();
    updateSimulationParam();

    //addFluidVolumes();

    //std::unique_ptr<ParticleObject> po = std::make_unique<ParticleObject>();
    //LoadBox(po.get(), cfloat3(-1.1, -0.015, -1.1),
    //    cfloat3(1.1, 1.1, 1.1),
    //    h_param.spacing);
    ////LoadBoundaryParticles(po.get());
    //float volfrac[] = { 0,0,0 };
    //addParticles(po->pos.size(), po->pos.data(), volfrac, GROUP_FIXED, TYPE_RIGID);
    //printf("%d boundary particles loaded.\n\n", po->pos.size());

    /*ParticleObject* duck = new ParticleObject();
        loadPoints(duck, "../resource/gear_test.obj");
        printf("duck has %d points\n", duck->pos.size());
        loadScriptObject(duck);
        delete duck;*/

    num_particles = pos.size();
}

void MultiphaseSPHSolver::loadSceneFromFile(char* filePath)
{
    SetParameter();
    updateSimulationParam();
    printf("time step: %f ms\n", h_param.dt);

    loadSimulationData(filePath, IDENTITY_MAT);
    num_particles = pos.size();
}

void MultiphaseSPHSolver::addFluidVolumes()
{
    for (int i = 0; i < fluid_volumes.size(); i++)
    {
        auto& fv  = *fluid_volumes[i];
        float pad = h_param.spacing * 0.5f;

        float spacing = h_param.spacing;

        cfloat3      xmin  = fv.xmin;
        cfloat3      xmax  = fv.xmax;
        const float* vf    = fv.volfrac;
        int          type  = fv.type;
        int          group = fv.group;

        xmax += cfloat3(pad, pad, pad);

        std::vector<cfloat3> p;
        for (float x = xmin.x; x < xmax.x; x += spacing)
            for (float y = xmin.y; y < xmax.y; y += spacing)
                for (float z = xmin.z; z < xmax.z; z += spacing)
                {
                    p.push_back(cfloat3(x, y, z));
                }
        addParticles(p.size(), p.data(), vf, group, type, 1);
    }
}

void MultiphaseSPHSolver::addParticles(int addcount, const cfloat3 add_pos[], const float volfrac[], int group_, int type_, int visible)
{
    if (type_ == TYPE_FLUID)
    {
        printf("Block type: fluid, particle num: %d\n", addcount);
        h_param.numFluidParticles += addcount;
    }
    else if (type_ == TYPE_DEFORMABLE)
    {
        printf("Block type: deformable, particle num: %d\n", addcount);
        h_param.numSolidParticles += addcount;
    }
    else if (type_ == TYPE_GRANULAR)
    {
        printf("Block type: granular, particle num: %d\n", addcount);
    }

    for (int ix = 0; ix < addcount; ix++)
    {
        int pid = addDefaultParticle();
        if (pid < 0)
        {
            printf("error: reaching particle number limit\n");
            return;
        }

        pos[pid]   = add_pos[ix];
        color[pid] = cfloat4(volfrac[0], volfrac[1], volfrac[2], visible);
        for (int t = 0; t < h_param.numTypes; t++)
            vol_frac[pid * h_param.numTypes + t] = volfrac[t];

        type[pid]  = type_;
        group[pid] = group_;
    }
    num_particles = pos.size();
}

void MultiphaseSPHSolver::SetParameter()
{

    h_param.gravity.set(0.0f, -9.8f, 0.0f);
    h_param.gridxmin.set(-1.2f, -0.5f, -1.2f);
    h_param.gridxmax.set(1.2f, 2.7f, 1.2f);

    h_param.dt           = 0.0005f;
    h_param.spacing      = 0.01f;
    float smoothratio    = 2.0f;
    h_param.smoothradius = h_param.spacing * smoothratio;
    h_param.viscosity    = 1.0f;
    h_param.restdensity  = 1000.0f;

    h_param.pressureK     = 100.0f;
    h_param.boundary_visc = 15.0f;

    h_param.dragForce  = 1;
    h_param.solid_visc = 5;

    h_param.numTypes   = 2;
    h_param.densArr[0] = 1000;
    h_param.densArr[1] = 1330;

    h_param.drift_dynamic_diffusion = 1;
    float padding                   = h_param.spacing * 1.5f;

    float width = 0.5;
    //auto fv = std::make_shared<fluidvol>();
    // SCENE1
    //fv->xmin.set(-1.2 + padding, 0.015, -0.5);
    //fv->xmax.set(-1.2 + padding + width, 1., 0.5);
    // SCENE2
    //fv->xmin.set(-1.2 + padding, 0.0, -0.5);
    //fv->xmax.set(0, 0.6, 0.5);
    //fv->group = 0;
    //fv->type = TYPE_FLUID;
    //fv->volfrac[0] = 1;
    //fv->volfrac[1] = 0;
    //fluid_volumes.push_back(fv);

    //fv = std::make_shared<fluidvol>();
    // SCENE1
    //fv->xmin.set(1.2 - padding - width, 0.015, -0.5);
    //fv->xmax.set(1.2 - padding, 1., 0.5);
    // SCENE2
    //fv->xmin.set(0.2, 0., -0.2);
    //fv->xmax.set(0.8, 0.45, 0.2);
    //fv->group = 0;
    //fv->type = TYPE_GRANULAR;
    //fv->volfrac[0] = 0;
    //fv->volfrac[1] = 1;
    //fluid_volumes.push_back(fv);

    h_param.solidG           = 100000;
    h_param.solidK           = 100000;
    h_param.granularFriction = 0.8;
    h_param.cohesion         = 0;

    h_param.dissolution = 1;
    max_nump            = 2500000;
}

void MultiphaseSPHSolver::updateSimulationParam()
{
    h_param.dx        = h_param.smoothradius;
    h_param.gridres.x = roundf((h_param.gridxmax.x - h_param.gridxmin.x) / h_param.dx);
    h_param.gridres.y = roundf((h_param.gridxmax.y - h_param.gridxmin.y) / h_param.dx);
    h_param.gridres.z = roundf((h_param.gridxmax.z - h_param.gridxmin.z) / h_param.dx);

    h_param.vol0        = h_param.spacing * h_param.spacing * h_param.spacing;
    h_param.melt_point  = 60;
    h_param.latent_heat = 10;

    enable_dfsph       = false;
    enable_solid       = true;
    enable_diffusion   = true;
    enable_heat        = false;
    enable_dump        = false;
    enable_dump_render = false;

    emit_timer    = 1.0f;
    emit_interval = 0.03f;
    dump_timer    = 0;
    frame_count   = 0;
    dump_count    = 0;

    h_param.enable_dfsph     = enable_dfsph;
    h_param.enable_solid     = enable_solid;
    h_param.enable_diffusion = enable_diffusion;
    h_param.enable_heat      = enable_heat;

    for (int i = 0; i < 100; i++)
        localid_counter[i] = 0;
}

void MultiphaseSPHSolver::dumpRenderData()
{
    printf("Saving render data: frame %d\n", frame_count);

    char filepath[1000];
    sprintf(filepath, "%s%d.txt", outputDir.c_str(), dump_count++);
    FILE* fp = fopen(filepath, "w+");
    if (fp == NULL)
    {
        printf("error opening file\n");
        return;
    }

    copySimulationDataFromDevice();

    fprintf(fp, "frame %d\n", frame_count);
    int output_count = 0;

    for (int i = 0; i < num_particles; i++)
    {
        /*if(type[i]!=TYPE_FLUID)
            continue;*/
        fprintf(fp, "%d %f %f %f ", output_count++, pos[i].x, pos[i].y, pos[i].z);
        fprintf(fp, "%f %f %f %d", vol_frac[i * h_param.numTypes], vol_frac[i * h_param.numTypes + 1], vol_frac[i * h_param.numTypes + 2], type[i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void MultiphaseSPHSolver::loadSimulationData(char* filepath, const cmat4& materialMat)
{
    printf("Loading simulation data text from %s", filepath);
    FILE* fp = fopen(filepath, "r");
    if (fp == NULL)
    {
        printf("error opening file\n");
        exit(-1);
    }
    auto vol = h_param.spacing * h_param.spacing * h_param.spacing;
    fscanf(fp, "%d\n", &num_particles);
    for (int pi = 0; pi < num_particles; pi++)
    {
        int  i     = addDefaultParticle();
        auto vFrac = vol_frac.data() + i * h_param.numTypes;

        fscanf3(fp, pos[i]);
        fscanf3(fp, vel[i]);
        //fscanf3(fp, normal[i]);
        fscanf(fp, "%d ", &type[i]);
        fscanf(fp, "%d ", &group[i]);
        //fscanf(fp, "%d ", &unique_id[i]);
        //fscanf3(fp, drift_accel[i]);
        if (type[i] != TYPE_RIGID)
        {
            fscanfn(fp, vFrac, h_param.numTypes);
            fscanfn(fp, stress[i].data, 9);
        }
        else
        {  //rigid color
            if (pos[i].y < 0.2)
                color[i] = cfloat4(1, 1, 1, 0.2);
            else
                color[i] = cfloat4(1, 1, 1, 0.0);
        }
        fscanf(fp, "\n");
    }
    fclose(fp);
}

void MultiphaseSPHSolver::dumpSimulationData()
{
    printf("Dumping simulation data at frame %d\n", frame_count);
    char filepath[100];
    sprintf(filepath, "dump/%d.txt", frame_count);
    FILE* fp = fopen(filepath, "w+");
    if (fp == NULL)
    {
        printf("error opening file\n");
        return;
    }
    copySimulationDataFromDevice();
    // Particle Data
    fprintf(fp, "%d\n", num_particles);
    for (int i = 0; i < num_particles; i++)
    {
        fprintf3(fp, pos[i]);
        fprintf3(fp, vel[i]);
        //fprintf3(fp, normal[i]);
        fprintf(fp, "%d ", type[i]);
        fprintf(fp, "%d ", group[i]);
        //fprintf(fp, "%d ", unique_id[i]);
        //fprintf3(fp, drift_accel[i]);
        if (type[i] != TYPE_RIGID)
        {
            fprintfn(fp, &vol_frac[i * h_param.numTypes], h_param.numTypes);
            fprintfn(fp, stress[i].data, 9);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int MultiphaseSPHSolver::addDefaultParticle()
{
    if (pos.size() == max_nump)
    {
        printf("Warning: Max numP reached.\n");
        return -1;
    }

    pos.push_back(cfloat3(0, 0, 0));
    color.push_back(cfloat4(1, 1, 1, 1));
    normal.push_back(cfloat3(0, 0, 0));
    unique_id.push_back(pos.size() - 1);
    vel.push_back(cfloat3(0, 0, 0));
    type.push_back(TYPE_FLUID);
    mass.push_back(0);
    rest_density.push_back(0);
    group.push_back(0);
    drift_accel.push_back(cfloat3(0, 0, 0));
    //localid.push_back(0);
    //temperature.push_back(0);
    //heat_buffer.push_back(0);

    for (int t = 0; t < h_param.numTypes; t++)
        vol_frac.push_back(0);
    if (enable_solid)
    {
        cmat3 zero_mat3;
        zero_mat3.Set(0.0f);
        stress.emplace_back(zero_mat3);
    }
    return pos.size() - 1;
}

inline bool outside(cfloat3& x, cfloat3& lowbound, cfloat3& upbound)
{
    return x.x < lowbound.x || x.y < lowbound.y || x.z < lowbound.z || x.x > upbound.x || x.y > upbound.y || x.z > upbound.z;
}

void MultiphaseSPHSolver::LoadBoundaryParticles(ParticleObject* po)
{
    for (int i = 0; i < po->pos.size(); i++)
    {
        int pid  = addDefaultParticle();
        pos[pid] = po->pos[i];
        /*if(pos[pid].y < 0.2 && pos[pid].z < 0)
                color[pid] = cfloat4(1, 1, 1, 0.2);
            else*/
        color[pid] = cfloat4(1, 1, 1, 0.0);
        type[pid]  = TYPE_RIGID;
        //normal[pid] = po->normal[i];
        group[pid] = GROUP_FIXED;
    }
}

void MultiphaseSPHSolver::LoadRigidParticles(ParticleObject* po,
                                             int             groupId)
{
    for (int i = 0; i < po->pos.size(); i++)
    {
        int pid     = addDefaultParticle();
        pos[pid]    = po->pos[i];
        color[pid]  = cfloat4(1, 1, 1, 0.2);
        type[pid]   = TYPE_RIGID;
        normal[pid] = po->normal[i];
        group[pid]  = groupId;
    }
}

void MultiphaseSPHSolver::loadScriptObject(ParticleObject* po)
{
    int addcount   = 0;
    int objectType = TYPE_RIGID;

    for (int i = 0; i < po->pos.size(); i++)
    {
        int pid  = addDefaultParticle();
        pos[pid] = po->pos[i];
        //pos[pid].y += 0.5;
        color[pid].set(1, 1, 1, 1);

        type[pid]  = TYPE_RIGID;
        group[pid] = GROUP_MOVABLE;
        addcount++;
    }
}

void MultiphaseSPHSolver::LoadPO(ParticleObject* po, int objectType)
{
    float vf[3]    = { 1, 0, 0 };
    int   addcount = 0;
    float spacing  = h_param.spacing;
    float density  = 0;
    for (int t = 0; t < h_param.numTypes; t++)
        density += h_param.densArr[t] * vf[t];

    //float relax = 1.65;
    for (int i = 0; i < po->pos.size(); i++)
    {
        int pid   = addDefaultParticle();
        pos[pid]  = po->pos[i];
        type[pid] = objectType;
        for (int t = 0; t < h_param.numTypes; t++)
            vol_frac[pid * h_param.numTypes + t] = vf[t];
        addcount++;
    }
}

void MultiphaseSPHSolver::emitFluid()
{
    if (emit_timer < emit_interval)
    {
        emit_timer += h_param.dt;
        return;
    }
    int nump_old = num_particles;

    emitFluidDisk(0.05f, cfloat3(0, 0.2, 0), cfloat3(0, 0, 0), cfloat3(0, -1, 0));
    //emitFluidDisk(0.05f, cfloat3(0, 0.41, 0), cfloat3(0, 0, 0), cfloat3(0, -1, 0));

    copyData2Device(nump_old, num_particles);
    updateMass_host(num_particles);
    emit_timer = 0;
}

void MultiphaseSPHSolver::emitFluidDisk(float radius, cfloat3 center, cfloat3 xyz, cfloat3 vp)
{

    float spacing = h_param.spacing;
    //float radius = 5.0f * spacing;
    float pad   = 0.5f * spacing;
    float vf[3] = { 1, 0, 0 };
    for (float xx = -radius; xx < radius + pad; xx += spacing)
        for (float zz = -radius; zz < radius + pad; zz += spacing)
        {
            cfloat3 xp = cfloat3(xx, 0, zz);
            RotateXYZ(xp, xyz);
            xp += center;
            int pid    = addDefaultParticle();
            pos[pid]   = xp;
            color[pid] = cfloat4(vf[0], vf[1], vf[2], 0.5f);
            type[pid]  = TYPE_DEFORMABLE;
            group[pid] = 0;
            vel[pid]   = vp;
            for (int t = 0; t < h_param.numTypes; t++)
                vol_frac[pid * h_param.numTypes + t] = vf[t];
            num_particles++;
        }
}

//============================================
//
// Computation
//
//============================================

void MultiphaseSPHSolver::sortParticles(uchar flag)
{
    computeParticleHash_host(
        simdata.pos,
        simdata.particleHash,
        simdata.particleIndex,
        h_param.gridxmin,
        h_param.dx,
        h_param.gridres,
        num_particles);

    sortParticleHash(
        simdata.particleHash,
        simdata.particleIndex,
        num_particles);

    findCellStart_host(
        simdata.particleHash,
        simdata.gridCellStart,
        simdata.gridCellEnd,
        num_particles,
        num_cells);

    if (flag == sortFlagsWithReorder)
    {
        reorder_by_handlers_host(bufman.d_sort_handlers,
                                 simdata.particleIndex,
                                 bufman.h_sort_handlers.size(),
                                 num_particles);
        bufman.copySortedData(num_particles);
    }
}

void MultiphaseSPHSolver::phaseDiffusion()
{
    computeDriftVel_host(num_particles);
    computeInterPhaseTensor_host(num_particles);

    computePhaseDiffusion_host(simdata, num_particles, h_param.numTypes);
    updateMass_host(num_particles);
}

void MultiphaseSPHSolver::solveWCSPH()
{

    //math::cTime clock;

#ifdef BUILD_NEIGHBOR_LIST
    BuildNeighborListHost(num_particles);
#endif
    //clock.tick();
    UpdateParticleStateHost(num_particles);
    //printf("p %f\n", clock.tack() * 1000);

    //clock.tick();
    computeForce_host(num_particles);
    //printf("f %f\n", clock.tack() * 1000);

    advectParticles_host(num_particles);

    if (stepCounter % 100 == 0)
        sortParticles(sortFlagsWithReorder);
    else
        sortParticles(sortFlagsWithoutReorder);
}

void MultiphaseSPHSolver::step()
{
    //math::cTime cp;
    //cp.tick();
    solveWCSPH();
    //printf("total %f\n", cp.tack() * 1000);

    stepCounter++;
    system_time += h_param.dt;
}

};  // namespace msph
