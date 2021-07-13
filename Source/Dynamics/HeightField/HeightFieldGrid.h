#pragma once
//#include "Framework/Framework/ModuleTopology.h"
//#include "Framework/Topology/NeighborList.h"
#include "Core/DataTypes.h"
#include "Core/Vector.h"
#include "Core/Array/Array2D.h"
#include "Core/Utility/Function1Pt.h"

namespace PhysIKA {
template <typename T, typename TReal = Real, DeviceType deviceType = DeviceType::GPU>
class HeightFieldGrid : public Array2D<T, deviceType>
{

public:
    //typedef typename TDataType::T T;
    typedef Vector<TReal, 3> Coord;

    HeightFieldGrid()
        :  //m_dx(0), m_dy(0),
        origin(0, 0, 0)
    {
        m_dx = 0;
        m_dz = 0;
    }

    ~HeightFieldGrid() {}

    //void copyFrom(HeightFieldGrid<TDataType>& pointSet);

    template <DeviceType sdeviceType = DeviceType::GPU>
    inline void set(const T* data, int dataPitch, int nx, int ny, float dx, float dy, float ox, float oy, float oz)
    {
        this->setSpace(dx, dy);
        this->setOrigin(ox, oy, oz);
        this->resize(nx, ny);

        // Copy data.
        if (deviceType == DeviceType::GPU && sdeviceType == DeviceType::GPU)
            (cudaMemcpy2D(this->GetDataPtr(), this->Pitch() * sizeof(T), data, dataPitch * sizeof(T), nx * sizeof(T), ny, cudaMemcpyDeviceToDevice));
        else if (deviceType == DeviceType::CPU && sdeviceType == DeviceType::GPU)
            (cudaMemcpy2D(this->GetDataPtr(), this->Pitch() * sizeof(T), data, dataPitch * sizeof(T), nx * sizeof(T), ny, cudaMemcpyDeviceToHost));
        else if (deviceType == DeviceType::GPU && sdeviceType == DeviceType::CPU)
            (cudaMemcpy2D(this->GetDataPtr(), this->Pitch() * sizeof(T), data, dataPitch * sizeof(T), nx * sizeof(T), ny, cudaMemcpyHostToDevice));
        else if (deviceType == DeviceType::CPU && sdeviceType == DeviceType::CPU)
            (cudaMemcpy2D(this->GetDataPtr(), this->Pitch() * sizeof(T), data, dataPitch * sizeof(T), nx * sizeof(T), ny, cudaMemcpyHostToHost));
    }

    inline void setOrigin(float ox, float oy, float oz)
    {
        origin[0] = ox;
        origin[1] = oy;
        origin[2] = oz;
    }

    inline void setSpace(TReal dx, TReal dz)
    {
        m_dx = dx;
        m_dz = dz;
    }

    COMM_FUNC inline int Nz()
    {
        return Ny();
    }

    COMM_FUNC inline TReal getDx()
    {
        return m_dx;
    }
    COMM_FUNC inline TReal getDz()
    {
        return m_dz;
    }

    COMM_FUNC inline Coord getOrigin()
    {
        return origin;
    }

    COMM_FUNC inline Coord gridPosition(int i, int j)
    {
        return Coord((i - Nx() / 2.0 + 0.5) * m_dx + origin[0],
                     origin[1],
                     (j - Ny() / 2.0 + 0.5) * m_dz + origin[2]);
    }

    /**
        * @brief Get index of a position on the grid. Returen unconstraint grid index.
        */
    COMM_FUNC inline int2 gridRawIndex(Coord pos)
    {
        pos = pos - origin;
        int2 res;
        res.x = pos[0] / m_dx + Nx() / 2.0;
        res.y = pos[2] / m_dz + Ny() / 2.0;
        return res;
    }

    /**
        * @brief Get index of a position on the grid. Constraint the index within grid index range.
        */
    COMM_FUNC inline int2 gridIndex(Coord pos)
    {
        int2 res = this->gridRawIndex(pos);
        res.x    = min(res.x, Nx() - 1);
        res.x    = max(res.x, 0);
        res.y    = min(res.y, Ny() - 1);
        res.y    = max(res.y, 0);
        return res;
    }

    /**
        * @brief Get center position of a grid (i,j).
        */
    COMM_FUNC inline Coord gridCenterPosition(int i, int j) const
    {
        return Coord((i - Nx() / 2.0 + 0.5f) * m_dx, 0.0, (j - Ny() / 2.0 + 0.5f) * m_dz) + origin;
    }

    /**
        * @brief Check validation of grid index.
        */
    COMM_FUNC inline bool inRange(int i, int j) const
    {
        return i >= 0 && i < Nx() && j >= 0 && j < Ny();
    }

    /**
        * @brief Find intersection between a triangle and grid center line
        * @details vt = t1 * v1 + t2* v2;  (t1,t2>=0,  t1+t2<=1)
        */
    COMM_FUNC inline bool onTopofGrid(int i, int j, const Coord& p0, const Coord& p1, const Coord& p2, Coord& res)
    {
        Coord gridcenter = gridCenterPosition(i, j);
        TReal v1x = p1[0] - p0[0], v1z = p1[2] - p0[2];
        TReal v2x = p2[0] - p0[0], v2z = p2[2] - p0[2];
        TReal vtx = gridcenter[0] - p0[0], vtz = gridcenter[2] - p0[2];
        TReal detv = v1x * v2z - v1z * v2x;
        TReal len  = sqrt(v1x * v1x + v1z * v1z + v2x * v2x + v2z * v2z);
        if (detv > -1e-5 * len && detv < 1e-5 * len)
            return false;
        TReal t1 = (v2z * vtx - v2x * vtz) / detv;
        TReal t2 = (v1x * vtz - v1z * vtx) / detv;
        if (t1 >= 0 && t2 >= 0 && t1 + t2 <= 1.0)
        {
            res = (1.0 - t1 - t2) * p0 + t1 * p1 + t2 * p2;
            return true;
        }
        res[0] = detv;
        res[1] = gridcenter[0], res[2] = gridcenter[2];
        return false;
    }

    COMM_FUNC inline T get(TReal x, TReal z)
    {
        TReal gx  = (x - origin[0]) / m_dx + Nx() / 2.0 - 0.5;
        TReal gz  = (z - origin[2]) / m_dz + Ny() / 2.0 - 0.5;
        int   gix = ( int )gx, giz = ( int )gz;
        gix         = min(gix, Nx() - 1);
        gix         = max(gix, 0);
        giz         = min(giz, Ny() - 1);
        giz         = max(giz, 0);
        int gix_1   = gix + 1;
        gix_1       = min(gix_1, Nx() - 1);
        int giz_1   = giz + 1;
        giz_1       = min(giz_1, Ny() - 1);
        TReal fracx = gx - gix, fracz = gz - giz;

        T val00 = (*this)(gix, giz);
        T val10 = (*this)(gix_1, giz);
        T val01 = (*this)(gix, giz_1);
        T val11 = (*this)(gix_1, giz_1);

        return val00 * ((1.0 - fracx) * (1.0 - fracz)) + val01 * ((1.0 - fracx) * fracz) + val10 * (fracx * (1.0 - fracz)) + val11 * (fracx * fracz);
    }

    COMM_FUNC inline T& safeGet(int i, int j)
    {
        i = max(i, 0);
        i = min(i, Nx() - 1);
        j = max(j, 0);
        j = min(j, Ny() - 1);
        return (*this)(i, j);
    }

    COMM_FUNC inline T safeGet(int i, int j) const
    {
        i = max(i, 0);
        i = min(i, Nx() - 1);
        j = max(j, 0);
        j = min(j, Ny() - 1);
        return (*this)(i, j);
    }

    COMM_FUNC inline void gradient(TReal x, TReal z, T& dhdx, T& dhdz)
    {
        /*    TReal gx = (x - origin[0]) / m_dx + Nx() / 2.0;
            TReal gz = (z - origin[2]) / m_dz + Ny() / 2.0;

            gx = min(gx, Nx() - 1.0);    gx = max(gx, 0.0);
            gz = min(gz, Ny() - 1.0);    gz = max(gz, 0.0);

            int gix = (int)(gx + 0.5), giz = (int)(gz + 0.5);
            int gix_1 = gix - 1, giz_1 = giz - 1;
            gix_1 = max(gix_1, 0);  gix_1 = min(gix_1, Nx() - 1);
            giz_1 = max(giz_1, 0);  giz_1 = min(giz_1, Ny() - 1);

            int gix_2 = gix + 1, giz_2 = giz + 1;
            gix_2 = max(gix_2, 0);  gix_2 = min(gix_2, Nx() - 1);
            giz_2 = max(giz_2, 0);  giz_2 = min(giz_2, Ny() - 1);

            TReal fracx = (gx - gix_1) / (double)(gix_2 - gix_1);
            TReal fracz = (gz - giz_1) / (double)(giz_2 - giz_1);

            T val00 = (*this)(gix_1, giz_1);
            T val10 = (*this)(gix_2, giz_1);
            T val01 = (*this)(gix_1, giz_2);
            T val11 = (*this)(gix_2, giz_2);
            dhdx = ((1.0 - fracz) * (val10 - val00) + fracz * (val11 - val01)) / (m_dx*(gix_2 - gix_1));
            dhdz = ((1.0 - fracx) * (val01 - val00) + fracx * (val11 - val10)) / (m_dz*(giz_2 - giz_1));
        */
        TReal gx = (x - origin[0]) / m_dx + Nx() / 2.0 - 0.5;
        TReal gz = (z - origin[2]) / m_dz + Ny() / 2.0 - 0.5;

        gx = min(gx, Nx() - 1.0);
        gx = max(gx, 0.0);
        gz = min(gz, Ny() - 1.0);
        gz = max(gz, 0.0);

        int gix = ( int )(gx), giz = ( int )(gz);
        int gix_1 = gix - 1, giz_1 = giz - 1;
        gix_1 = max(gix_1, 0);
        gix_1 = min(gix_1, Nx() - 1);
        giz_1 = max(giz_1, 0);
        giz_1 = min(giz_1, Ny() - 1);

        int gix_2 = gix + 1, giz_2 = giz + 1;
        gix_2 = max(gix_2, 0);
        gix_2 = min(gix_2, Nx() - 1);
        giz_2 = max(giz_2, 0);
        giz_2 = min(giz_2, Ny() - 1);

        TReal fracx = (gx - gix_1) / ( double )(gix_2 - gix_1);
        TReal fracz = (gz - giz_1) / ( double )(giz_2 - giz_1);

        T val00 = (*this)(gix_1, giz_1);
        T val10 = (*this)(gix_2, giz_1);
        T val01 = (*this)(gix_1, giz_2);
        T val11 = (*this)(gix_2, giz_2);
        dhdx    = ((1.0 - fracz) * (val10 - val00) + fracz * (val11 - val01)) / (m_dx * (gix_2 - gix_1));
        dhdz    = ((1.0 - fracx) * (val01 - val00) + fracx * (val11 - val10)) / (m_dz * (giz_2 - giz_1));

        ////if (gix>=38&& gix<=41 && giz>=28 && giz<=32)
        //if(print)
        //    printf("   %d %d, %lf.  Detail:  %lf   %d %d %d %d,  %lf %lf %lf %lf \n",
        //        gix, giz, (*this)(gix, giz),
        //        dhdx, gix_1, giz_1, gix_2, giz_2,
        //        val00, val10, val01, val11);
    }

    /**
        * @brief Calculate normal vector of a point in height field. 
        * 
        * @details normal = (-dh/dx, 1, -dh/dz).
        * @details Type T should support +/-* operators.
        */
    COMM_FUNC inline Coord heightFieldNormal(TReal x, TReal z)
    {
        T dhdx = 0, dhdz = 0;
        gradient(x, z, dhdx, dhdz);
        return Coord(-dhdx, 1.0, -dhdz).normalize();
    }

    //COMM_FUNC inline DeviceArray2D<T>& getGrid() { return m_grid; }

protected:
    //         DeviceArray<Coord> m_coords;
    //         DeviceArray<Coord> m_normals;
    //         NeighborList<int> m_pointNeighbors;

    Coord origin;

    TReal m_dx;
    TReal m_dz;

    //Array2D<T, deviceType> m_grid;
};

template <typename T>
using HostGrid2D = HeightFieldGrid<T, Real, DeviceType::CPU>;

template <typename T>
using DeviceGrid2D = HeightFieldGrid<T, Real, DeviceType::GPU>;

using DeviceGrid2Df = DeviceGrid2D<float>;
using HostGrid2Df   = HostGrid2D<float>;

using DeviceHeightField1f = HeightFieldGrid<float, float, DeviceType::GPU>;
using HostHeightField1f   = HeightFieldGrid<float, float, DeviceType::CPU>;
using DeviceHeightField1d = HeightFieldGrid<double, double, DeviceType::GPU>;
using HostHeightField1d   = HeightFieldGrid<double, double, DeviceType::CPU>;
using DeviceHeightField3f = HeightFieldGrid<Vector3f, float, DeviceType::GPU>;
using HostHeightField3f   = HeightFieldGrid<Vector3f, float, DeviceType::CPU>;
using DeviceHeightField3d = HeightFieldGrid<Vector3d, double, DeviceType::GPU>;
using HostHeightField3d   = HeightFieldGrid<Vector3d, double, DeviceType::CPU>;

#ifdef PRECISION_FLOAT
template class HeightFieldGrid<float, float, DeviceType::GPU>;
template class HeightFieldGrid<float, float, DeviceType::CPU>;
#else
template class HeightFieldGrid<DataType3d>;
#endif
}  // namespace PhysIKA
