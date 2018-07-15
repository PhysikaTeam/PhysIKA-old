#include <fstream>
#include "DistanceField3D.h"
#include "Physika_Core/Utilities/cuda_helper_math.h"

namespace Physika{

	template <typename Coord>
	__device__  float DistanceToPlane(const Coord &p, const Coord &o, const Coord &n) {
		return fabs((p - o, n).length());
	}

	template <typename Coord>
	__device__  Real DistanceToSegment(Coord& pos, Coord& lo, Coord& hi)
	{
		typedef typename Coord::VarType Real;
		Coord seg = hi - lo;
		Coord edge1 = pos - lo;
		Coord edge2 = pos - hi;
		if (edge1.dot(seg) < 0.0f)
		{
			return edge1.norm();
		}
		if (edge2.dot(-seg) < 0.0f)
		{
			return edge2.norm();
		}
		Real length1 = edge1.dot(edge1);
		seg.normalize();
		Real length2 = edge1.dot(seg);
		return std::sqrt(length1 - length2*length2);
	}

	template <typename Coord>
	__device__  Real DistanceToSqure(Coord& pos, Coord& lo, Coord& hi, int axis)
	{
		typedef typename Coord::VarType Real;
		Coord n;
		Coord corner1, corner2, corner3, corner4;
		Coord loCorner, hiCorner, p;
		switch (axis)
		{
		case 0:
			corner1 = Coord(lo[0], lo[1], lo[2]);
			corner2 = Coord(lo[0], hi[1], lo[2]);
			corner3 = Coord(lo[0], hi[1], hi[2]);
			corner4 = Coord(lo[0], lo[1], hi[2]);
			n = Coord(1.0, 0.0, 0.0);

			loCorner = Coord(lo[1], lo[2], 0.0);
			hiCorner = Coord(hi[1], hi[2], 0.0);
			p = Coord(pos[1], pos[2], 0.0f);
			break;
		case 1:
			corner1 = Coord(lo[0], lo[1], lo[2]);
			corner2 = Coord(lo[0], lo[1], hi[2]);
			corner3 = Coord(hi[0], lo[1], hi[2]);
			corner4 = Coord(hi[0], lo[1], lo[2]);
			n = Coord(0.0f, 1.0f, 0.0f);

			loCorner = Coord(lo[0], lo[2], 0.0f);
			hiCorner = Coord(hi[0], hi[2], 0.0f);
			p = Coord(pos[0], pos[2], 0.0f);
			break;
		case 2:
			corner1 = Coord(lo[0], lo[1], lo[2]);
			corner2 = Coord(hi[0], lo[1], lo[2]);
			corner3 = Coord(hi[0], hi[1], lo[2]);
			corner4 = Coord(lo[0], hi[1], lo[2]);
			n = Coord(0.0f, 0.0f, 1.0f);

			loCorner = Coord(lo[0], lo[1], 0.0);
			hiCorner = Coord(hi[0], hi[1], 0.0);
			p = Coord(pos[0], pos[1], 0.0f);
			break;
		}

		Real dist1 = DistanceToSegment(pos, corner1, corner2);
		Real dist2 = DistanceToSegment(pos, corner2, corner3);
		Real dist3 = DistanceToSegment(pos, corner3, corner4);
		Real dist4 = DistanceToSegment(pos, corner4, corner1);
		Real dist5 = abs(n.dot(pos - corner1));
		if (p[0] < hiCorner[0] && p[0] > loCorner[0] && p[1] < hiCorner[1] && p[1] > loCorner[1])
			return dist5;
		else
			return min(min(dist1, dist2), min(dist3, dist4));
	}

	template <typename Coord>
	__device__  Real DistanceToBox(Coord& pos, Coord& lo, Coord& hi)
	{
		typedef typename Coord::VarType Real;
		Coord corner0(lo[0], lo[1], lo[2]);
		Coord corner1(hi[0], lo[1], lo[2]);
		Coord corner2(hi[0], hi[1], lo[2]);
		Coord corner3(lo[0], hi[1], lo[2]);
		Coord corner4(lo[0], lo[1], hi[2]);
		Coord corner5(hi[0], lo[1], hi[2]);
		Coord corner6(hi[0], hi[1], hi[2]);
		Coord corner7(lo[0], hi[1], hi[2]);
		Real dist0 = (pos - corner0).norm();
		Real dist1 = (pos - corner1).norm();
		Real dist2 = (pos - corner2).norm();
		Real dist3 = (pos - corner3).norm();
		Real dist4 = (pos - corner4).norm();
		Real dist5 = (pos - corner5).norm();
		Real dist6 = (pos - corner6).norm();
		Real dist7 = (pos - corner7).norm();
		if (pos[0] < hi[0] && pos[0] > lo[0] && pos[1] < hi[1] && pos[1] > lo[1] && pos[2] < hi[2] && pos[2] > lo[2])
		{
			Real distx = min(abs(pos[0] - hi[0]), abs(pos[0] - lo[0]));
			Real disty = min(abs(pos[1] - hi[1]), abs(pos[1] - lo[1]));
			Real distz = min(abs(pos[2] - hi[2]), abs(pos[2] - lo[2]));
			Real mindist = min(distx, disty);
			mindist = min(mindist, distz);
			return mindist;
		}
		else
		{
			Real distx1 = DistanceToSqure(pos, corner0, corner7, 0);
			Real distx2 = DistanceToSqure(pos, corner1, corner6, 0);
			Real disty1 = DistanceToSqure(pos, corner0, corner5, 1);
			Real disty2 = DistanceToSqure(pos, corner3, corner6, 1);
			Real distz1 = DistanceToSqure(pos, corner0, corner2, 2);
			Real distz2 = DistanceToSqure(pos, corner4, corner6, 2);
			return -min(min(min(distx1, distx2), min(disty1, disty2)), min(distz1, distz2));
		}
	}

	template <typename Real, typename Coord>
	__device__  Real DistanceToCylinder(Coord& pos, Coord& center, Real radius, Real height, int axis)
	{
		Real distR;
		Real distH;
		switch (axis)
		{
		case 0:
			distH = abs(pos[0] - center[0]);
			distR = Coord(0.0, pos[1] - center[1], pos[2] - center[2]).norm();
			break;
		case 1:
			distH = abs(pos[1] - center[1]);
			distR = Coord(pos[0] - center[0], 0.0, pos[2] - center[2]).norm();
			break;
		case 2:
			distH = abs(pos[2] - center[2]);
			distR = Coord(pos[0] - center[0], pos[1] - center[1], 0.0).norm();
			break;
		}

		Real halfH = height / 2.0f;
		if (distH <= halfH && distR <= radius)
		{
			return -min(halfH - distH, radius - distR);
		}
		else if (distH > halfH && distR <= radius)
		{
			return distH - halfH;
		}
		else if (distH <= halfH && distR > radius)
		{
			return distR - radius;
		}
		else
		{
// 			Real l1 = distR - radius;
// 			Real l2 = distH - halfH;
//			return sqrt(l1*l1 + l2*l2);
			return Vector<Real, 2>(distR - radius, distH - halfH).norm();
		}


	}

	template <typename Real, typename Coord>
	__device__  Real DistanceToSphere(Coord& pos, Coord& center, Real radius)
	{
		return (pos - center).length() - radius;
	}

	template<typename TDataType>
	DistanceField3D<TDataType>::DistanceField3D()
	{
	}

	template<typename TDataType>
	DistanceField3D<TDataType>::DistanceField3D(std::string filename)
	{
		ReadSDF(filename);
		bInvert = false;
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::SetSpace(const Coord p0, const Coord p1, int nbx, int nby, int nbz)
	{
		left = p0;

		h = (p1 - p0)*Coord(1.0 / Real(nbx+1), 1.0 / Real(nby+1), 1.0 / Real(nbz+1));

		gDist.Resize(nbx+1, nby+1, nbz+1);
	}

	template<typename TDataType>
	DistanceField3D<TDataType>::~DistanceField3D()
	{
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::Translate(const Coord &t) {
		left += t;
	}

	template <typename Real>
	__global__ void K_Scale(DeviceArray3D<Real> distance, float s)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.Nx()) return;
		if (j >= distance.Ny()) return;
		if (k >= distance.Nz()) return;

		distance(i, j, k) = s*distance(i, j, k);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::Scale(const Real s) {
		left[0] *= s;
		left[1] *= s;
		left[2] *= s;
		h[0] *= s;
		h[1] *= s;
		h[2] *= s;

		dim3 blockSize = make_uint3(8, 8, 8);
		dim3 gridDims = cudaGridSize3D(make_uint3(gDist.Nx(), gDist.Ny(), gDist.Nz()), blockSize);

		K_Scale << <gridDims, blockSize >> >(gDist, s);
	}

	template<typename Real>
	__global__ void K_Invert(DeviceArray3D<Real> distance)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.Nx()) return;
		if (j >= distance.Ny()) return;
		if (k >= distance.Nz()) return;

		distance(i, j, k) = -distance(i, j, k);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::Invert()
	{
		dim3 blockSize = make_uint3(8, 8, 8);
		dim3 gridDims = cudaGridSize3D(make_uint3(gDist.Nx(), gDist.Ny(), gDist.Nz()), blockSize);

		K_Invert << <gridDims, blockSize >> >(gDist);
	}

	template <typename Real, typename Coord>
	__global__ void K_DistanceFieldToBox(DeviceArray3D<Real> distance, Coord start, Coord h, Coord lo, Coord hi, bool inverted)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.Nx()) return;
		if (j >= distance.Ny()) return;
		if (k >= distance.Nz()) return;

		int sign = inverted ? 1.0f : -1.0f;
		Coord p = start + Coord(i, j, k)*h;

		distance(i, j, k) = sign*DistanceToBox(p, lo, hi);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::DistanceFieldToBox(Coord& lo, Coord& hi, bool inverted)
	{
		bInvert = inverted;

		dim3 blockSize = make_uint3(4, 4, 4);
		dim3 gridDims = cudaGridSize3D(make_uint3(gDist.Nx(), gDist.Ny(), gDist.Nz()), blockSize);

		K_DistanceFieldToBox << <gridDims, blockSize >> >(gDist, left, h, lo, hi, inverted);
	}

	template <typename Real, typename Coord>
	__global__ void K_DistanceFieldToCylinder(DeviceArray3D<Real> distance, Coord start, Coord h, Coord center, Real radius, Real height, int axis, bool inverted)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.Nx()) return;
		if (j >= distance.Ny()) return;
		if (k >= distance.Nz()) return;

		int sign = inverted ? -1.0f : 1.0f;

		Coord p = start + Coord(i, j, k)*h;

		distance(i, j, k) = sign*DistanceToCylinder(p, center, radius, height, axis);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::DistanceFieldToCylinder(Coord& center, Real radius, Real height, int axis, bool inverted)
	{
		bInvert = inverted;

		dim3 blockSize = make_uint3(8, 8, 8);
		dim3 gridDims = cudaGridSize3D(make_uint3(gDist.Nx(), gDist.Ny(), gDist.Nz()), blockSize);

		K_DistanceFieldToCylinder << <gridDims, blockSize >> >(gDist, left, h, center, radius, height, axis, inverted);
	}

	template <typename Real, typename Coord>
	__global__ void K_DistanceFieldToSphere(DeviceArray3D<Real> distance, Coord start, Coord h, Coord center, Real radius, bool inverted)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.Nx()) return;
		if (j >= distance.Ny()) return;
		if (k >= distance.Nz()) return;

		int sign = inverted ? -1.0f : 1.0f;

		Coord p = start + Coord(i, j, k)*h;

		Coord dir = p - center;

		distance(i, j, k) = sign*(dir.norm()-radius);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::DistanceFieldToSphere(Coord& center, Real radius, bool inverted)
	{
		bInvert = inverted;

		dim3 blockSize = make_uint3(8, 8, 8);
		dim3 gridDims = cudaGridSize3D(make_uint3(gDist.Nx(), gDist.Ny(), gDist.Nz()), blockSize);

		K_DistanceFieldToSphere << <gridDims, blockSize >> >(gDist, left, h, center, radius, inverted);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::ReadSDF(std::string filename)
	{
		std::ifstream input(filename.c_str(), std::ios::in);
		int nbx, nby, nbz;
		int xx, yy, zz;
		input >> xx;
		input >> yy;
		input >> zz;
		input >> left[0];
		input >> left[1];
		input >> left[2];
		Real t_h;
		input >> t_h;

		std::cout << "SDF: " << xx << ", " << yy << ", " << zz << std::endl;
		std::cout << "SDF: " << left[0] << ", " << left[1] << ", " << left[2] << std::endl;
		std::cout << "SDF: " << left[0] + t_h*xx << ", " << left[1] + t_h*yy << ", " << left[2] + t_h*zz << std::endl;

		nbx = xx;
		nby = yy;
		nbz = zz;
		h[0] = t_h;
		h[1] = t_h;
		h[2] = t_h;

		int idd = 0;
		Real* distances = new Real[(nbx)*(nby)*(nbz)];
		for (int k = 0; k < zz; k++) {
			for (int j = 0; j < yy; j++) {
				for (int i = 0; i < xx; i++) {
					float dist;
					input >> dist;
					distances[i + nbx*(j + nby*k)] = dist;
				}
			}
		}
		input.close();

		gDist.Resize(nbx, nby, nbz);
		cudaCheck(cudaMemcpy(gDist.GetDataPtr(), distances, (nbx)*(nby)*(nbz) * sizeof(Real), cudaMemcpyHostToDevice));

		std::cout << "read data successful" << std::endl;
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::Release()
	{
		gDist.Release();
	}
}