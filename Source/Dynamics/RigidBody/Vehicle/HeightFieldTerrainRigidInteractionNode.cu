//#include "HeightFieldTerrainRigidInteractionNode.h"
#include "Dynamics/RigidBody/Vehicle/HeightFieldTerrainRigidInteractionNode.h"
//#include <cuda_runtime.h>
//#include <curand_kernel.h>
#include "Framework/Framework/ModuleTopology.h"
#include "Dynamics/Sand/types.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "Dynamics/RigidBody/RigidTimeIntegrationModule.h"
#include "Dynamics/RigidBody/RKIntegrator.h"

#include <iostream>

#include "thrust/reduce.h"
#include "thrust/execution_policy.h"
#include "thrust/device_vector.h"

namespace PhysIKA {
HeightFieldTerrainRigidInteractionNode::~HeightFieldTerrainRigidInteractionNode()
{
}

bool HeightFieldTerrainRigidInteractionNode::initialize()
{

    //terrainInfo.elasticModulus = 5e6;
    //terrainInfo.surfaceThickness = 0.05;

    return true;
}

void HeightFieldTerrainRigidInteractionNode::advance(Real dt)
{
    _test(dt);
    //auto allRigids = m_rigidSystem->getAllNode();
    //for (auto prigid : allRigids)
    //{
    //	// Calculate interaction force and apply to rigid.
    //	Vector3f force, torque;
    //	if (this->_calcualteSingleContactForce(prigid, force, torque))
    //	{
    //		prigid->addExternalForce(force);
    //		prigid->addExternalTorque(torque);
    //	}
    //}
}

void HeightFieldTerrainRigidInteractionNode::setSize(int nx, int ny)
{
    if (m_nx * m_ny != nx * ny)
    {
        m_forceSummator = std::shared_ptr<Reduction<Vector3f>>(Reduction<Vector3f>::Create(nx * ny));
        //m_maxDepthFinder = std::shared_ptr<Reduction<ContactPointX>>(Reduction<ContactPointX>::Create(nx * ny));
    }

    m_nx = nx;
    m_ny = ny;

    m_heightField.resize(nx, ny);
    m_interactForce.resize(nx * ny);
    m_interactTorque.resize(nx * ny);
}

void HeightFieldTerrainRigidInteractionNode::setSize(int nx, int ny, float dx, float dy)
{
    this->setSize(nx, ny);
    m_heightField.setSpace(dx, dy);
}

__global__ void calculateGridForce(DeviceGrid2Df                         grid,
                                   DeviceArray<Vector3f>                 gridforce,
                                   DeviceArray<Vector3f>                 gridtorque,
                                   DeviceArray<TopologyModule::Triangle> faces,
                                   DeviceArray<Vector3f>                 vertices,
                                   RigidInteractionInfo                  rigidinfo,
                                   TerrainRigidInteractionInfo           interactInfo)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= faces.size())
    {
        return;
    }

    //printf("%d  \n", tid);

    auto     face       = faces[tid];
    Vector3f p0         = rigidinfo.rotation.rotate(vertices[face[0]]) + rigidinfo.position;
    Vector3f p1         = rigidinfo.rotation.rotate(vertices[face[1]]) + rigidinfo.position;
    Vector3f p2         = rigidinfo.rotation.rotate(vertices[face[2]]) + rigidinfo.position;
    Vector3f faceNormal = (p1 - p0).cross(p2 - p0).normalize();

    float xmin = p0[0], ymin = p0[1], zmin = p0[2];
    float xmax = p0[0], ymax = p0[1], zmax = p0[2];
    xmin = p1[0] < xmin ? p1[0] : xmin;
    ymin = p1[1] < ymin ? p1[1] : ymin;
    zmin = p1[2] < zmin ? p1[2] : zmin;
    xmax = p1[0] > xmax ? p1[0] : xmax;
    ymax = p1[1] > ymax ? p1[1] : ymax;
    zmax = p1[2] > zmax ? p1[2] : zmax;
    xmin = p2[0] < xmin ? p2[0] : xmin;
    ymin = p2[1] < ymin ? p2[1] : ymin;
    zmin = p2[2] < zmin ? p2[2] : zmin;
    xmax = p2[0] > xmax ? p2[0] : xmax;
    ymax = p2[1] > ymax ? p2[1] : ymax;
    zmax = p2[2] > zmax ? p2[2] : zmax;

    // Find potential intersection grids.
    int2 gmin = grid.gridIndex(Vector3f(xmin, ymin, zmin));
    int2 gmax = grid.gridIndex(Vector3f(xmax, ymax, zmax));

    // For each grid, find intersection.
    for (int i = gmin.x; i <= gmax.x; ++i)
    {
        for (int j = gmin.y; j <= gmax.y; ++j)
        {
            Vector3f hitPoint;
            bool     intersect = grid.onTopofGrid(i, j, p0, p1, p2, hitPoint);

            if (intersect)
            {
                float depth = grid(i, j) - hitPoint[1];

                depth = fmaxf(0, depth);
                depth = fminf(depth, interactInfo.surfaceThickness);

                float area     = /*fmaxf(0, -faceNormal[1])* */ grid.getDx() * grid.getDz();
                float pressure = interactInfo.elasticModulus * depth / interactInfo.surfaceThickness
                                 + interactInfo.damping * rigidinfo.linearVelocity.dot(faceNormal);

                //if (depth > 1e-4)
                //	printf("%d: %d  %d;    %f\n", tid, i, j, pressure);

                //grid2Dwrite(gridforce.begin(), i, j, grid.Nx(), -pressure * area * faceNormal);
                Vector3f force                = -pressure * area * faceNormal;
                gridforce[j * grid.Nx() + i]  = force;
                gridtorque[j * grid.Nx() + i] = (hitPoint - rigidinfo.position).cross(force);
            }
        }
    }
}

bool HeightFieldTerrainRigidInteractionNode::_calcualteSingleContactForce(std::shared_ptr<RigidBody2<DataType3f>> prigid, Vector3f& force, Vector3f& torque)
{
    //dim3 bdim(16, 16, 1);
    //dim3 gdim = cudaGridSize3D(dim3(m_heightField.Nx(), m_heightField.Ny(), 1), bdim);
    auto triangleset = TypeInfo::cast<TriangleSet<DataType3f>>(prigid->getTopologyModule());
    if (!triangleset)
        return false;

    DeviceArray<TopologyModule::Triangle>* triangles = triangleset->getTriangles();
    DeviceArray<Vector3f>&                 vertices  = triangleset->getPoints();

    RigidInteractionInfo rigidinfo;
    rigidinfo.position        = prigid->getGlobalR();
    rigidinfo.rotation        = prigid->getGlobalQ();
    rigidinfo.linearVelocity  = prigid->getLinearVelocity();
    rigidinfo.angularVelocity = prigid->getAngularVelocity();

    m_interactForce.reset();
    m_interactTorque.reset();

    cudaDeviceSynchronize();
    //cudaThreadSynchronize();
    //cudaError_t err = cudaGetLastError();

    int bdim = 16 * 16;
    int gdim = cudaGridSize(triangles->size(), bdim);

    calculateGridForce<<<gdim, bdim>>>(m_heightField, m_interactForce, m_interactTorque, *triangles, vertices, rigidinfo, terrainInfo);
    //cudaDeviceSynchronize();
    cudaThreadSynchronize();
    //cudaError_t err2 = cudaGetLastError();
    cuSynchronize();

    force  = m_forceSummator->accumulate(m_interactForce.begin(), m_nx * m_ny);
    torque = m_forceSummator->accumulate(m_interactTorque.begin(), m_nx * m_ny);

    std::cout << " -----  Force: " << force[0] << "   " << force[1] << "   " << force[2] << std::endl;

    return true;
}

COMM_FUNC long long int toLongLong(float depth, int i, int j)
{
    long long int res = depth * 1e4;
    res               = res * 1000000;
    res += i * 1000 + j;  // i and j is less then 1000
    return res;
}

COMM_FUNC void fromLongLong(long long int lldepth, float& depth, int& i, int& j)
{
    j = lldepth % 1000;
    lldepth /= 1000;
    i = lldepth % 1000;
    lldepth /= 1000;
    depth = ( float )lldepth / 10000.0;
}

__global__ void calculateDepthPerRigidPoint(DeviceGrid2Df              grid,
                                            DeviceArray<ContactPointX> gridDepth,
                                            //DeviceArray<TopologyModule::Triangle> faces,
                                            DeviceArray<Vector3f> vertices,
                                            RigidInteractionInfo  rigidinfo)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= vertices.size())
    {
        return;
    }

    //printf("%d  \n", tid);

    auto vertex = rigidinfo.rotation.rotate(vertices[tid]) + rigidinfo.position;

    float h     = grid.get(vertex[0], vertex[2]);
    float depth = h - vertex[1];

    if (depth >= 0)
    {

        Vector3f pointnormal = grid.heightFieldNormal(vertex[0], vertex[2]);
        //int2 gidx = grid.gridIndex(vertex);

        //printf("%d:  %f   %f   %f\n", tid, pointnormal[0], pointnormal[1], pointnormal[2]);

        ContactPointX cp;
        cp.depth       = depth;
        cp.point       = make_float3(vertex[0], vertex[1], vertex[2]);
        cp.normal      = make_float3(-pointnormal[0], -pointnormal[1], -pointnormal[2]);
        gridDepth[tid] = cp;
    }
    else
    {
        gridDepth[tid].depth = -1;
    }
}

__global__ void calculateDepth(DeviceGrid2Df                         grid,
                               DeviceArray<ContactPointX>            gridDepth,
                               DeviceArray<TopologyModule::Triangle> faces,
                               DeviceArray<Vector3f>                 vertices,
                               RigidInteractionInfo                  rigidinfo)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= faces.size())
    {
        return;
    }

    //printf("%d  \n", tid);

    auto     face       = faces[tid];
    Vector3f p0         = rigidinfo.rotation.rotate(vertices[face[0]]) + rigidinfo.position;
    Vector3f p1         = rigidinfo.rotation.rotate(vertices[face[1]]) + rigidinfo.position;
    Vector3f p2         = rigidinfo.rotation.rotate(vertices[face[2]]) + rigidinfo.position;
    Vector3f faceNormal = (p1 - p0).cross(p2 - p0).normalize();

    float xmin = p0[0], ymin = p0[1], zmin = p0[2];
    float xmax = p0[0], ymax = p0[1], zmax = p0[2];
    xmin = p1[0] < xmin ? p1[0] : xmin;
    ymin = p1[1] < ymin ? p1[1] : ymin;
    zmin = p1[2] < zmin ? p1[2] : zmin;
    xmax = p1[0] > xmax ? p1[0] : xmax;
    ymax = p1[1] > ymax ? p1[1] : ymax;
    zmax = p1[2] > zmax ? p1[2] : zmax;
    xmin = p2[0] < xmin ? p2[0] : xmin;
    ymin = p2[1] < ymin ? p2[1] : ymin;
    zmin = p2[2] < zmin ? p2[2] : zmin;
    xmax = p2[0] > xmax ? p2[0] : xmax;
    ymax = p2[1] > ymax ? p2[1] : ymax;
    zmax = p2[2] > zmax ? p2[2] : zmax;

    // Find potential intersection grids.
    int2 gmin = grid.gridIndex(Vector3f(xmin, ymin, zmin));
    int2 gmax = grid.gridIndex(Vector3f(xmax, ymax, zmax));

    // For each grid, find intersection.
    for (int i = gmin.x; i <= gmax.x; ++i)
    {
        for (int j = gmin.y; j <= gmax.y; ++j)
        {
            Vector3f hitPoint;
            bool     intersect = grid.onTopofGrid(i, j, p0, p1, p2, hitPoint);

            if (intersect)
            {
                float depth = grid(i, j) - hitPoint[1];
                if (depth >= 0)
                {

                    ContactPointX cp;
                    cp.depth                     = depth;
                    cp.point                     = make_float3(hitPoint[0], hitPoint[1], hitPoint[2]);
                    cp.normal                    = make_float3(faceNormal[0], faceNormal[1], faceNormal[2]);
                    gridDepth[j * grid.Nx() + i] = cp;
                }
            }
        }
    }
}

float _contactPointRelV(const SystemMotionState& mState, const ContactPointX& cPoint, int id)
{
    Vector3f relp = Vector3f(cPoint.point.x, cPoint.point.y, cPoint.point.z) - mState.globalPosition[id];

    Transform3d<float>   movetobody(mState.globalPosition[id], Quaternion<float>(0, 0, 0, 1));
    SpatialVector<float> vel = movetobody.transformM(mState.globalVelocity[id]);
    Vector3f             linv(vel[3], vel[4], vel[5]);
    Vector3f             angv(vel[0], vel[1], vel[2]);
    Vector3f             newv = angv.cross(relp) + linv;
    return -newv.dot(Vector3f(cPoint.normal.x, cPoint.normal.y, cPoint.normal.z));
}

bool HeightFieldTerrainRigidInteractionNode::_test(Real dt)
{

    //thrust::device_vector<ContactPointX> depth;

    //depth[0];

    auto allRigids = m_rigidSystem->getAllNode();
    auto fdsolver  = m_rigidSystem->getModule<RigidTimeIntegrationModule>();
    m_rigidSystem->collectForceState();  // Update external force.

    // Set free system state.
    auto        ps0 = m_rigidSystem->getSystemState();
    SystemState state;
    state = *ps0;
    //for (int i=0;i<state.m_externalForce.size();++i)
    //{
    //	state.m_externalForce[i] = SpatialVector<float>();
    //}
    //for (int i = 0; i < state.m_activeForce.size(); ++i)
    //{
    //	state.m_activeForce[i] = 0;
    //}

    Vectornd<float> ddq(m_rigidSystem->getJointDof());
    //state.setNum(allRigids.size(), m_rigidSystem->getJointDof());

    DSystemMotionState dstate;
    SystemMotionState  motionstate;

    std::vector<std::pair<int, ContactPointX>> contactPoints(solveFriction ? 3 * allRigids.size() : allRigids.size());
    int                                        contactN = 0;

    // Contact detection between rigid body and height field.
    for (int i = 0; i < allRigids.size(); ++i)
    {
        auto prigid      = allRigids[i];
        auto triangleset = TypeInfo::cast<TriangleSet<DataType3f>>(prigid->getTopologyModule());
        if (!triangleset)
            continue;

        ContactPointX maxdep;

        DeviceArray<TopologyModule::Triangle>* triangles = triangleset->getTriangles();
        DeviceArray<Vector3f>&                 vertices  = triangleset->getPoints();

        RigidInteractionInfo rigidinfo;
        rigidinfo.position        = prigid->getGlobalR();
        rigidinfo.rotation        = prigid->getGlobalQ();
        rigidinfo.linearVelocity  = prigid->getLinearVelocity();
        rigidinfo.angularVelocity = prigid->getAngularVelocity();

        cudaDeviceSynchronize();
        //cudaThreadSynchronize();
        //cudaError_t err = cudaGetLastError();

        if (detectionMethod == HFDETECTION::FACEVISE)
        {
            m_depthScratch.resize(m_heightField.Nx() * m_heightField.Ny());
            m_depthScratch.reset();

            int bdim = 16 * 16;
            int gdim = cudaGridSize(triangles->size(), bdim);

            calculateDepth<<<gdim, bdim>>>(m_heightField, m_depthScratch, *triangles, vertices, rigidinfo);
            cudaThreadSynchronize();
            //cudaError_t err2 = cudaGetLastError();
            cuSynchronize();

            maxdep = thrust::reduce(thrust::device, m_depthScratch.begin(), m_depthScratch.begin() + m_nx * m_ny, ContactPointX(), thrust::maximum<ContactPointX>());
        }
        else
        {
            int npoint = vertices.size();
            m_depthScratch.resize(npoint);
            m_depthScratch.reset();

            int bdim = 16 * 16;
            int gdim = cudaGridSize(npoint, bdim);

            calculateDepthPerRigidPoint<<<gdim, bdim>>>(m_heightField, m_depthScratch, vertices, rigidinfo);
            cudaThreadSynchronize();
            //cudaError_t err2 = cudaGetLastError();
            cuSynchronize();

            maxdep = thrust::reduce(thrust::device, m_depthScratch.begin(), m_depthScratch.begin() + npoint, ContactPointX(), thrust::maximum<ContactPointX>());
        }

        //ContactPointX maxdep = m_maxDepthFinder->maximum(depth.begin(), m_nx * m_ny);

        if (maxdep.depth > 0)
        {
            contactPoints[contactN++] = std::make_pair(i, maxdep);

            if (solveFriction)
            {
                int           fnid      = contactN - 1;
                ContactPointX contactt1 = maxdep;
                contactt1.beFriction    = fnid;
                ContactPointX contactt2 = maxdep;
                contactt2.beFriction    = fnid;

                Vector3f curnormal(maxdep.normal.x, maxdep.normal.y, maxdep.normal.z);
                Vector3f dirt1(0, 0, -1);
                dirt1          = rigidinfo.rotation.rotate(dirt1);
                Vector3f dirt2 = curnormal.cross(dirt1).normalize();
                dirt1          = dirt2.cross(curnormal).normalize();

                contactt1.normal = make_float3(dirt1[0], dirt1[1], dirt1[2]);
                contactt2.normal = make_float3(dirt2[0], dirt2[1], dirt2[2]);

                contactPoints[contactN++] = std::make_pair(i, contactt1);
                contactPoints[contactN++] = std::make_pair(i, contactt2);
            }
        }
    }

    if (contactN == 0)
        return true;
    m_dantzigInput.resize(contactN);

    // b = - free state rigid velocity.
    fdsolver->dydt(state, *(state.m_motionState), dstate);
    //RK4Integrator rk4;
    //rk4.solve(*(s.m_motionState), DydtAdapter(fdsolver.get()), dt);
    motionstate = *(state.m_motionState);
    motionstate.addDs(dstate, dt);
    for (int i = 0; i < contactN; ++i)
    {
        int            id    = contactPoints[i].first;
        ContactPointX& contP = contactPoints[i].second;

        m_dantzigInput.b[i] = -_contactPointRelV(motionstate, contP, id);
    }

    //m_dantzigInput.resize(contactN);
    for (int i = 0; i < contactN; ++i)
    {
        int            rigidi = contactPoints[i].first;
        ContactPointX& contP  = contactPoints[i].second;
        Vector3f       contpoint(contP.point.x, contP.point.y, contP.point.z);
        Vector3f       contnormal(contP.normal.x, contP.normal.y, contP.normal.z);

        // Relative position of contact point.
        Vector3f pos  = state.m_motionState->globalPosition[rigidi];
        Vector3f relp = contpoint - pos;

        // Set test force to 1.
        Vector3f             testf       = -/*10**/ contnormal;
        Vector3f             testtor     = relp.cross(testf);
        SpatialVector<float> extForceBef = state.m_externalForce[rigidi];
        state.m_externalForce[rigidi] += SpatialVector<float>(testtor, testf);

        // Solve test force.
        fdsolver->dydt(state, *(state.m_motionState), dstate);
        motionstate = *(state.m_motionState);
        motionstate.addDs(dstate, dt);

        // Reset force.
        state.m_externalForce[rigidi] = extForceBef;

        // Matrix A(i,j) = w(fi=1, j) + b(j)
        for (int j = 0; j < contactN; ++j)
        {
            float an                           = _contactPointRelV(motionstate, contactPoints[j].second, contactPoints[j].first);
            m_dantzigInput.A[j * contactN + i] = an + m_dantzigInput.b[j];

            if (i == j)
                m_dantzigInput.A[j * contactN + i] *= 1.01;
        }
    }

    //for (int i = 0; i < contactN; ++i)
    //{
    //	for (int j = 0; j < contactN ; ++j)
    //	{
    //		if((i==3||j==3) && i!=j)
    //			m_dantzigInput.A[j*contactN + i] = 0;
    //
    //	}

    //}

    // lo(i) = 0, hi(i) = max
    for (int i = 0; i < contactN; ++i)
    {
        if (contactPoints[i].second.beFriction < 0)
        {
            m_dantzigInput.lo[i]     = 0;
            m_dantzigInput.hi[i]     = BT_INFINITY;
            m_dantzigInput.findex[i] = -1;
        }
        else
        {
            m_dantzigInput.hi[i]     = mu;
            m_dantzigInput.lo[i]     = mu;
            m_dantzigInput.findex[i] = contactPoints[i].second.beFriction;
        }
    }

    // Solve LCP.
    bool succ = btSolveDantzigLCP(contactN, m_dantzigInput.A.begin(), m_dantzigInput.x.begin(), m_dantzigInput.b.begin(), m_dantzigInput.w.begin(), 0, m_dantzigInput.lo.begin(), m_dantzigInput.hi.begin(), m_dantzigInput.findex.begin(), m_dantzigscratch);

    if (succ)
    {
        for (int i = 0; i < contactN; ++i)
        {
            int            rigidi = contactPoints[i].first;
            ContactPointX& contP  = contactPoints[i].second;
            Vector3f       contpoint(contP.point.x, contP.point.y, contP.point.z);
            Vector3f       contnormal(contP.normal.x, contP.normal.y, contP.normal.z);

            auto prigid = allRigids[rigidi];

            // Relative position of contact point.
            Vector3f pos  = state.m_motionState->globalPosition[rigidi];
            Vector3f relp = contpoint - pos;

            // Set test force to 1.
            Vector3f testf   = (-m_dantzigInput.x[i] /* * 10*/) * contnormal;
            Vector3f testtor = relp.cross(testf);

            prigid->addExternalForce(testf);
            prigid->addExternalTorque(testtor);
        }
    }

    return succ;
}

}  // namespace PhysIKA