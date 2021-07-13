#include "PointSDFContactDetector.h"
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

namespace PhysIKA {
PointSDFContactDetector::PointSDFContactDetector()
{
}

__global__ void PSCD_detectContacts(
    ContactInfo<double>*             contacts,
    DeviceDArray<Vector3d>           points,
    DistanceField3D<DataType3f>      sdf,
    DeviceArray<PBDBodyInfo<double>> body,
    int                              sdfid,
    double                           maxdis = 0.0)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= points.size())
        return;

    Vector3d pointd = points[tid];
    body[sdfid].pose.invTransform(pointd);
    Vector3f pointf(pointd[0], pointd[1], pointd[2]);

    Vector3f normalf;
    float    dis;
    sdf.getDistance(pointf, dis, normalf);
    normalf *= -1.0;
    Vector3d normal(normalf[0], normalf[1], normalf[2]);

    body[sdfid].pose.rotate(normal);

    if (dis < maxdis)
    {
        //Vector3d tran = body[sdfid].pose.position;
        //printf("dis: %f %lf, point: %lf %lf %lf, %lf %lf %lf \n",dis, maxdis,
        //	points[tid][0], points[tid][1], points[tid][2],
        //	tran[0], tran[1], tran[2]
        //	);

        // contact.
        contacts[tid].id0    = sdfid;
        contacts[tid].id1    = tid;
        contacts[tid].normal = normal;
        contacts[tid].point0 = points[tid] - normal * dis;
        contacts[tid].point1 = points[tid];
    }
    else
    {
        contacts[tid].id0 = -1;
        contacts[tid].id1 = -1;
    }
}

struct _PSCD_noneContact
{
    COMM_FUNC bool operator()(const ContactInfo<double>& ci)
    {
        return ci.id0 < 0 || ci.id1 < 0;
    }
};
void PointSDFContactDetector::compute(DeviceDArray<ContactInfo<double>>& contacts,
                                      DeviceDArray<Vector3d>&            points,
                                      DistanceField3D<DataType3f>&       sdf,
                                      DeviceArray<PBDBodyInfo<double>>&  body,
                                      int                                sdfid,
                                      int                                beginidx)
{
    //
    int np = points.size();
    contacts.resize(beginidx + np);
    cuExecute(np, PSCD_detectContacts, contacts.begin() + beginidx, points, sdf, body, sdfid);

    auto iter = thrust::remove_if(thrust::device, contacts.begin() + beginidx, contacts.begin() + beginidx + np, _PSCD_noneContact());
    contacts.resize(iter - contacts.begin());

    //HostDArray<ContactInfo<double>> hostcontact;
    //hostcontact.resize(contacts.size());
    //Function1Pt::copy(hostcontact, contacts);

    //HostArray<PBDBodyInfo<double>> hostbody;
    //hostbody.resize(body.size());
    //Function1Pt::copy(hostbody, body);

    //hostcontact.release();
    //hostbody.release();
}

void PointMultiSDFContactDetector::compute()
{
    if (!m_contacts || !m_particlePos || !m_sdfs)
        return;
    m_contacts->resize(0);
    int n = m_sdfs->size();
    for (int i = 0; i < n; ++i)
    {
        m_singleSDFDetector.compute(*m_contacts,
                                    *m_particlePos,
                                    (*m_sdfs)[i],
                                    *m_body,
                                    i,
                                    m_contacts->size());
    }
}

}  // namespace PhysIKA
