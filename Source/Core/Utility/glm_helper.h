#ifndef GLM_HELPER_H
#define GLM_HELPER_H
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>

inline __host__ __device__ glm::vec3 Float2Vec(float3& v)
{
    return glm::vec3(v.x, v.y, v.z);
}

inline __host__ __device__ float3 Vec2Float(glm::vec3& v)
{
    return make_float3(v.x, v.y, v.z);
}

inline __host__ __device__ void JacobiRotate(glm::mat3& A, glm::mat3& R, int p, int q)
{
    // rotates A through phi in pq-plane to set A(p,q) = 0
    // rotation stored in R whose columns are eigenvectors of A
    if (A[p][q] == 0.0f)
        return;

    float d = (A[p][p] - A[q][q]) / (2.0f * A[p][q]);
    float t = 1.0f / (fabs(d) + sqrt(d * d + 1.0f));
    if (d < 0.0f)
        t = -t;
    float c = 1.0f / sqrt(t * t + 1);
    float s = t * c;
    A[p][p] += t * A[p][q];
    A[q][q] -= t * A[p][q];
    A[p][q] = 0.0f;
    A[q][p] = 0.0f;
    // transform A
    int k;
    for (k = 0; k < 3; k++)
    {
        if (k != p && k != q)
        {
            float Akp = c * A[k][p] + s * A[k][q];
            float Akq = -s * A[k][p] + c * A[k][q];
            A[k][p]   = Akp;
            A[p][k]   = Akp;
            A[k][q]   = Akq;
            A[q][k]   = Akq;
        }
    }
    // store rotation in R
    for (k = 0; k < 3; k++)
    {
        float Rkp = c * R[k][p] + s * R[k][q];
        float Rkq = -s * R[k][p] + c * R[k][q];
        R[k][p]   = Rkp;
        R[k][q]   = Rkq;
    }
}

inline __host__ __device__ void EigenDecomposition(const glm::mat3& A, glm::mat3& eigenVecs, glm::vec3& eigenVals)
{
    const int   numJacobiIterations = 10;
    const float epsilon             = 1e-15f;

    glm::mat3 D = A;

    // only for symmetric Matrix!
    eigenVecs = glm::mat3();  // unit matrix
    int iter  = 0;
    while (iter < numJacobiIterations)
    {  // 3 off diagonal elements
        // find off diagonal element with maximum modulus
        int   p, q;
        float a, max;
        max = fabs(D[0][1]);
        p   = 0;
        q   = 1;
        a   = fabs(D[0][2]);
        if (a > max)
        {
            p   = 0;
            q   = 2;
            max = a;
        }
        a = fabs(D[1][2]);
        if (a > max)
        {
            p   = 1;
            q   = 2;
            max = a;
        }
        // all small enough -> done
        if (max < epsilon)
            break;
        // rotate matrix with respect to that element
        JacobiRotate(D, eigenVecs, p, q);
        iter++;
    }
    eigenVals[0] = D[0][0];
    eigenVals[1] = D[1][1];
    eigenVals[2] = D[2][2];
}

inline __host__ __device__ void PolarDecomposition(const glm::mat3& A, glm::mat3& R, glm::mat3& U, glm::mat3& D)
{
    // A = SR, where S is symmetric and R is orthonormal
    // -> S = (A A^T)^(1/2)

    // A = U D U^T R

    glm::mat3 AAT;
    AAT[0][0] = A[0][0] * A[0][0] + A[0][1] * A[0][1] + A[0][2] * A[0][2];
    AAT[1][1] = A[1][0] * A[1][0] + A[1][1] * A[1][1] + A[1][2] * A[1][2];
    AAT[2][2] = A[2][0] * A[2][0] + A[2][1] * A[2][1] + A[2][2] * A[2][2];

    AAT[0][1] = A[0][0] * A[1][0] + A[0][1] * A[1][1] + A[0][2] * A[1][2];
    AAT[0][2] = A[0][0] * A[2][0] + A[0][1] * A[2][1] + A[0][2] * A[2][2];
    AAT[1][2] = A[1][0] * A[2][0] + A[1][1] * A[2][1] + A[1][2] * A[2][2];

    AAT[1][0] = AAT[0][1];
    AAT[2][0] = AAT[0][2];
    AAT[2][1] = AAT[1][2];

    R = glm::mat3();
    glm::vec3 eigenVals;
    EigenDecomposition(AAT, U, eigenVals);

    float d0 = sqrt(eigenVals[0]);
    float d1 = sqrt(eigenVals[1]);
    float d2 = sqrt(eigenVals[2]);
    D        = glm::mat3(0.0f);
    D[0][0]  = d0;
    D[1][1]  = d1;
    D[2][2]  = d2;

    const float eps = 1e-15f;

    float l0 = eigenVals[0];
    if (l0 <= eps)
        l0 = 0.0f;
    else
        l0 = 1.0f / d0;
    float l1 = eigenVals[1];
    if (l1 <= eps)
        l1 = 0.0f;
    else
        l1 = 1.0f / d1;
    float l2 = eigenVals[2];
    if (l2 <= eps)
        l2 = 0.0f;
    else
        l2 = 1.0f / d2;

    glm::mat3 S1;
    S1[0][0] = l0 * U[0][0] * U[0][0] + l1 * U[0][1] * U[0][1] + l2 * U[0][2] * U[0][2];
    S1[1][1] = l0 * U[1][0] * U[1][0] + l1 * U[1][1] * U[1][1] + l2 * U[1][2] * U[1][2];
    S1[2][2] = l0 * U[2][0] * U[2][0] + l1 * U[2][1] * U[2][1] + l2 * U[2][2] * U[2][2];

    S1[0][1] = l0 * U[0][0] * U[1][0] + l1 * U[0][1] * U[1][1] + l2 * U[0][2] * U[1][2];
    S1[0][2] = l0 * U[0][0] * U[2][0] + l1 * U[0][1] * U[2][1] + l2 * U[0][2] * U[2][2];
    S1[1][2] = l0 * U[1][0] * U[2][0] + l1 * U[1][1] * U[2][1] + l2 * U[1][2] * U[2][2];

    S1[1][0] = S1[0][1];
    S1[2][0] = S1[0][2];
    S1[2][1] = S1[1][2];

    R = A * S1;

    // stabilize
    glm::vec3 c0, c1, c2;
    c0 = R[0];
    c1 = R[1];
    c2 = R[2];

    if (c0.length() < eps)
        c0 = glm::cross(c1, c2);
    else if (c1.length() < eps)
        c1 = glm::cross(c2, c0);
    else
        c2 = glm::cross(c0, c1);
    R[0] = c0;
    R[1] = c1;
    R[2] = c2;
}

inline __host__ __device__ float OneNorm(const glm::mat3& A)
{
    const float sum1   = fabs(A[0][0]) + fabs(A[1][0]) + fabs(A[2][0]);
    const float sum2   = fabs(A[0][1]) + fabs(A[1][1]) + fabs(A[2][1]);
    const float sum3   = fabs(A[0][2]) + fabs(A[1][2]) + fabs(A[2][2]);
    float       maxSum = sum1;
    if (sum2 > maxSum)
        maxSum = sum2;
    if (sum3 > maxSum)
        maxSum = sum3;
    return maxSum;
}

/** Return the inf norm of the matrix.
*/
inline __host__ __device__ float InfNorm(const glm::mat3& A)
{
    const float sum1   = fabs(A[0][0]) + fabs(A[1][0]) + fabs(A[2][0]);
    const float sum2   = fabs(A[0][1]) + fabs(A[1][1]) + fabs(A[2][1]);
    const float sum3   = fabs(A[0][2]) + fabs(A[1][2]) + fabs(A[2][2]);
    float       maxSum = sum1;
    if (sum2 > maxSum)
        maxSum = sum2;
    if (sum3 > maxSum)
        maxSum = sum3;
    return maxSum;
}

inline __host__ __device__ void PolarDecompositionStable(const glm::mat3& M, const float tolerance, glm::mat3& R)
{
    glm::mat3 Mt   = glm::transpose(M);
    float     Mone = OneNorm(M);
    float     Minf = InfNorm(M);
    float     Eone;
    glm::mat3 MadjTt, Et;
    do
    {
        MadjTt[0] = glm::cross(Mt[1], Mt[2]);
        MadjTt[1] = glm::cross(Mt[2], Mt[0]);
        MadjTt[2] = glm::cross(Mt[0], Mt[1]);

        float det = Mt[0][0] * MadjTt[0][0] + Mt[0][1] * MadjTt[0][1] + Mt[0][2] * MadjTt[0][2];

        if (fabs(det) < 1.0e-12)
        {
            glm::vec3    len;
            unsigned int index = 0xffffffff;
            for (unsigned int i = 0; i < 3; i++)
            {
                len[i] = MadjTt[i].length();
                if (len[i] > 1.0e-12)
                {
                    // index of valid cross product
                    // => is also the index of the vector in Mt that must be exchanged
                    index = i;
                    break;
                }
            }
            if (index == 0xffffffff)
            {
                R = glm::mat3();
                return;
            }
            else
            {
                Mt[index]               = glm::cross(Mt[(index + 1) % 3], Mt[(index + 2) % 3]);
                MadjTt[(index + 1) % 3] = glm::cross(Mt[(index + 2) % 3], Mt[(index) % 3]);
                ;
                MadjTt[(index + 2) % 3] = glm::cross(Mt[(index) % 3], Mt[(index + 1) % 3]);
                glm::mat3 M2            = glm::transpose(Mt);
                Mone                    = OneNorm(M2);
                Minf                    = InfNorm(M2);
                det                     = Mt[0][0] * MadjTt[0][0] + Mt[0][1] * MadjTt[0][1] + Mt[0][2] * MadjTt[0][2];
            }
        }

        const float MadjTone = OneNorm(MadjTt);
        const float MadjTinf = InfNorm(MadjTt);

        const float gamma = sqrt(sqrt((MadjTone * MadjTinf) / (Mone * Minf)) / fabs(det));

        const float g1 = gamma * 0.5f;
        const float g2 = 0.5f / (gamma * det);

        for (unsigned char i = 0; i < 3; i++)
        {
            for (unsigned char j = 0; j < 3; j++)
            {
                Et[i][j] = Mt[i][j];
                Mt[i][j] = g1 * Mt[i][j] + g2 * MadjTt[i][j];
                Et[i][j] -= Mt[i][j];
            }
        }

        Eone = OneNorm(Et);

        Mone = OneNorm(Mt);
        Minf = InfNorm(Mt);
    } while (Eone > Mone * tolerance);

    // Q = Mt^T
    R = glm::transpose(Mt);
}

#endif