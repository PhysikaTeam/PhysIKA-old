
#include "Dynamics/Sand/HeightFieldSandRigidInteraction.h"
#include "Dynamics/Sand/SSEUtil.h"
#include <functional>

namespace PhysIKA {
HeightFieldSandRigidInteraction::HeightFieldSandRigidInteraction()
{
    m_interactSolver           = std::make_shared<SandInteractionForceSolver>();
    m_landRigidContactDetector = std::make_shared<HeightFieldBodyDetector>();

    var_BouyancyFactor.setValue(300);
    var_DragFactor.setValue(2.0);
    var_CHorizontal.setValue(1.0);
    var_CVertical.setValue(1.0);
}

bool HeightFieldSandRigidInteraction::initialize()
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if (m_sandSolver)
    {
        m_psandInfo = m_sandSolver->getSandGridInfo();
    }
    if (m_psandInfo)
    {
        int nx  = m_psandInfo->nx;
        int ny  = m_psandInfo->ny;
        int nxy = nx * ny;

        m_gridParticle.resize(nxy);
        m_gridVel.resize(nxy);
        m_gridVel.reset();

        m_sandSolver->m_gridVel = &m_gridVel;
    }

    if (m_rigidSolver && m_psandInfo)
    {
        // Interaction solver.
        m_interactSolver->m_prigids          = &(m_rigidSolver->getRigidBodys());
        m_interactSolver->m_body             = &(m_rigidSolver->getGPUBody());
        m_interactSolver->m_hostBody         = &(m_rigidSolver->getCPUBody()[0]);
        m_interactSolver->m_particlePos      = &m_gridParticle;
        m_interactSolver->m_particleMass     = &m_gridMass;
        m_interactSolver->m_particleVel      = &m_gridVel;
        m_interactSolver->m_land             = m_landHeight;
        m_interactSolver->m_beAddForceUpdate = false;
        //m_interactSolver->m_useStickParticleVelUpdate = true;

        //
        m_interactSolver->m_smoothLength = 0;
        //m_interactSolver->m_neighbor = 0;

        m_interactSolver->m_CsHorizon      = var_CHorizontal.getValue();
        m_interactSolver->m_CsVertical     = var_CVertical.getValue();
        m_interactSolver->m_Cdrag          = var_DragFactor.getValue();
        m_interactSolver->m_buoyancyFactor = var_BouyancyFactor.getValue();
    }

    this->var_Contacts.setValue(DeviceDArray<ContactInfo<double>>());
    this->var_DetectThreshold.setValue(0.0);

    // Initialize contact detector
    if (m_landRigidContactDetector)
    {
        this->var_DetectThreshold.connect(m_landRigidContactDetector->varThreshold());
        this->var_Contacts.connect(m_landRigidContactDetector->varContacts());
        m_landRigidContactDetector->m_land = this->m_landHeight;

        if (m_rigidSolver)
        {
            // Init collision body ptr.
            auto& rigids = m_rigidSolver->getRigidBodys();
            for (auto prigid : rigids)
            {
                m_landRigidContactDetector->addCollidableObject(prigid);
            }

            // Set contact detection function.
            m_rigidSolver->setNarrowDetectionFun(std::bind(&HeightFieldSandRigidInteraction::detectLandRigidContacts,
                                                           this,
                                                           std::placeholders::_1,
                                                           std::placeholders::_2));
        }
    }

    return true;
}

void HeightFieldSandRigidInteraction::advance(Real dt)
{
    CTimer timer;
    timer.start();

    if (!m_sandSolver)
    {
        double subdt = dt / m_subStep;
        for (int i = 0; i < m_subStep; ++i)
        {
            this->advectSubStep(subdt);

            //float subdt2 = m_sandSolver->getMaxTimeStep();
            //printf("  Cur max time step:  %f\n", subdt2);
        }
    }
    else
    {
        do
        {
            double subdt = m_sandSolver->getMaxTimeStep();
            subdt        = subdt < dt ? subdt : dt;
            dt -= subdt;
            printf("  Cur time step:  %f\n", subdt);

            this->advectSubStep(subdt);

        } while (dt > 0);
    }

    timer.stop();
    double elapTime = timer.getElapsedTime();

    ++m_totalFrame;
    m_totalTime += elapTime;
    double avgTime = m_totalTime / m_totalFrame;
    printf("Elapsed time:  %lf,  Average time: %lf\n", elapTime, avgTime);
}

void HeightFieldSandRigidInteraction::advectSubStep(Real dt)
{

    // Advect sand.
    if (m_sandSolver)
    //if(false)
    {
        m_sandSolver->advection(dt);
        m_sandSolver->updateSandGridHeight();

        //m_sandSolver->stepSimulation(dt);
    }

    if (m_rigidSolver && m_sandSolver)
    //if(false)
    {
        m_interactSolver->setPreBodyInfo();
    }

    if (m_rigidSolver)
    {
        Real rdt = dt / m_subRigidStep;
        for (int i = 0; i < m_subRigidStep; ++i)
        {
            m_rigidSolver->forwardSubStepGPU(rdt);
        }
        m_rigidSolver->updateGPUToCPUBody();
    }

    if (m_rigidSolver && m_sandSolver)
    //if(false)
    {
        this->_updateSandHeightField();

        m_interactSolver->updateBodyAverageVel(dt);

        int nrigid = m_rigidSolver->getRigidBodys().size();
        for (int i = 0; i < nrigid; ++i)
        {
            if (m_interactSolver->collisionValid(m_rigidSolver->getRigidBodys()[i]))
            {
                // Update info.
                this->_updateGridParticleInfo(i);

                // Solve interaction force.
                m_interactSolver->computeSingleBody(i, dt);

                // Solver sand density constraint.
                m_sandSolver->applyVelocityChange(dt, m_minGi, m_minGj, m_sizeGi, m_sizeGj);
            }
        }

        m_rigidSolver->synFromBodiedToRigid();
    }

    if (m_sandSolver)
    {
        m_sandSolver->updateVeclocity(dt);

        //m_sandSolver->advection(dt);
        //m_sandSolver->updateSandGridHeight();

        //m_sandSolver->updateSandStaticHeight(dt);
    }
}

void HeightFieldSandRigidInteraction::setSandGrid(DeviceHeightField1d& sandHeight, DeviceHeightField1d& landHeight)
{
    m_sandHeight = &sandHeight;
    m_landHeight = &landHeight;
}

void HeightFieldSandRigidInteraction::detectLandRigidContacts(PBDSolver* solver, Real dt)
{
    if (m_landRigidContactDetector)
    {
        m_landRigidContactDetector->doCollision();

        auto& contactArr = m_landRigidContactDetector->varContacts()->getValue();
        m_rigidSolver->setContactJoints(contactArr,
                                        contactArr.size());
    }
}

void HeightFieldSandRigidInteraction::_setRigidForceAsGravity()
{
    if (!m_rigidSolver)
        return;

    auto& rigids = m_rigidSolver->getRigidBodys();
    for (auto prigid : rigids)
    {
        prigid->setExternalForce(Vector3f(0, -m_gravity * prigid->getI().getMass(), 0));
        prigid->setExternalTorque(Vector3f(0, 0, 0));
    }
}

void HeightFieldSandRigidInteraction::_setRigidForceEmpty()
{
    if (!m_rigidSolver)
        return;

    auto& rigids = m_rigidSolver->getRigidBodys();
    for (auto prigid : rigids)
    {
        prigid->setExternalForce(Vector3f(0, 0, 0));
        prigid->setExternalTorque(Vector3f(0, 0, 0));
    }
}

__global__ void HFSRI_updateGridParticleInfo(
    DeviceDArray<Vector3d> parPos,
    DeviceDArray<Vector3d> parVel,
    DeviceDArray<double>   parMass,
    SandGridInfo           sandinfo,
    DeviceHeightField1d    sandheight,
    DeviceHeightField1d    landheight,
    int                    minGx,
    int                    minGz,
    int                    sizeGx,
    int                    sizeGz)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= (sizeGx * sizeGz))
        return;

    int gi = tid % sizeGx + minGx;
    int gj = tid / sizeGx + minGz;

    Vector3d pos = sandheight.gridCenterPosition(gi, gj);
    pos[1]       = sandheight(gi, gj) + landheight(gi, gj);
    parPos[tid]  = pos;

    float4 gp   = grid2Dread(sandinfo.data, gi, gj, sandinfo.pitch);
    float  velu = SSEUtil::d_get_u(gp);
    float  velv = SSEUtil::d_get_v(gp);
    parVel[tid] = Vector3d(velv, 0.0, velu);

    parMass[tid] = sandinfo.sandRho * sandinfo.griddl * sandinfo.griddl * sandheight(gi, gj);
}

__global__ void HFSRI_updateHeightField(
    DeviceHeightField1d sandheight,
    DeviceHeightField1d landheight,
    SandGridInfo        sandinfo)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;

    if (tidx >= sandinfo.nx || tidy >= sandinfo.ny)
        return;
    gridpoint gp           = grid2Dread(sandinfo.data, tidx, tidy, sandinfo.pitch);
    sandheight(tidx, tidy) = gp.x;
    landheight(tidx, tidy) = gp.w;
}

void HeightFieldSandRigidInteraction::_updateSandHeightField()
{
    if (!m_psandInfo || !m_rigidSolver)
        return;

    uint3 gsize = { m_psandInfo->nx, m_psandInfo->ny, 1 };
    cuExecute2D(gsize, HFSRI_updateHeightField, *m_sandHeight, *m_landHeight, *m_psandInfo);
}

void HeightFieldSandRigidInteraction::_updateGridParticleInfo(int i)
{
    if (!m_psandInfo || !m_rigidSolver)
        return;

    auto prigid = m_rigidSolver->getRigidBodys()[i];
    if (!prigid)
        return;

    int minGx = 0, minGz = 0, maxGx = 0, maxGz = 0;
    this->_computeBoundingGrid(minGx, minGz, maxGx, maxGz, prigid->getRadius(), prigid->getGlobalR());

    m_minGi  = minGx;
    m_minGj  = minGz;
    m_sizeGi = maxGx - minGx;
    m_sizeGj = maxGz - minGz;

    uint3 gsize = { m_sizeGi, m_sizeGj, 1 };
    //cuExecute2D(gsize, HFSRI_updateHeightField,
    //	m_sandHeight, m_landHeight,
    //	*m_psandInfo
    //);

    //uint3 bDims = { 16,16,1 };
    //uint3 pDims = cudaGridSize3D(gsize, bDims);
    //dim3 threadsPerBlock(16, 16, 1);
    //HFSRI_updateHeightField << <pDims, threadsPerBlock >> > (
    //	*m_sandHeight, *m_landHeight,
    //	*m_psandInfo
    //);

    //cuSynchronize();

    int numPar = gsize.x * gsize.y;
    m_gridParticle.resize(numPar);
    m_gridVel.resize(numPar);
    m_gridMass.resize(numPar);

    cuExecute(gsize.x * gsize.y, HFSRI_updateGridParticleInfo, m_gridParticle, m_gridVel, m_gridMass, *m_psandInfo, *m_sandHeight, *m_landHeight, m_minGi, m_minGj, m_sizeGi, m_sizeGj);
}

void HeightFieldSandRigidInteraction::_computeBoundingGrid(int& minGx, int& minGz, int& maxGx, int& maxGz, float radius, const Vector3f& center)
{
    Vector3d minp(center[0] - radius, center[1], center[2] - radius);
    int2     minG = m_sandHeight->gridIndex(minp);
    minGx         = minG.x - 3;
    minGx         = max(minGx, 0);
    minGz         = minG.y - 3;
    minGz         = max(minGz, 0);

    Vector3d maxp(center[0] + radius, center[1], center[2] + radius);
    int2     maxG = m_sandHeight->gridIndex(maxp);
    maxGx         = maxG.x + 3;
    maxGx         = min(maxGx, m_sandHeight->Nx());
    maxGz         = maxG.y + 3;
    maxGz         = min(maxGz, m_sandHeight->Ny());
}
}  // namespace PhysIKA