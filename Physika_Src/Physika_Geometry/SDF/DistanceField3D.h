/******************************************************************************
Copyright (c) 2016 Xiaowei He (xiaowei@iscas.ac.com)

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
******************************************************************************/
#ifndef FRAMEWORK_DISTANCEFIELD3D_H
#define FRAMEWORK_DISTANCEFIELD3D_H

#include <string>
#include "Physika_Core/Platform.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Vectors/vector.h"
#include "Physika_Core/Cuda_Array/Array3D.h"
#include "Physika_Core/DataTypes.h"

namespace Physika {

	template<typename TDataType>
	class DistanceField3D {
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DistanceField3D();

		DistanceField3D(std::string filename);

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~DistanceField3D();

		void SetSpace(const Coord p0, const Coord p1, int nbx, int nby, int nbz);
		void Release();

		void Translate(const Coord& t);
		void Scale(const Real s);
		void Invert();

		void DistanceFieldToBox(Coord& lo, Coord& hi, bool inverted);

		void DistanceFieldToCylinder(Coord& center, Real radius, Real height, int axis, bool inverted);

		void DistanceFieldToSphere(Coord& center, Real radius, bool inverted);

		COMM_FUNC void GetDistance(const Coord &p, Real &d, Coord &g) {
			// get cell and lerp values
			Coord fp = (p - left)*Coord(1.0/h[0], 1.0/h[1], 1.0/h[2]);
			const int i = (int)floor(fp[0]);
			const int j = (int)floor(fp[1]);
			const int k = (int)floor(fp[2]);
			if (i < 0 || i >= gDist.Nx() - 1 || j < 0 || j >= gDist.Ny() - 1 || k < 0 || k >= gDist.Nz()-1) {
				if (bInvert) d = -100000.0f;
				else d = 100000.0f;
				g = Coord(0);
				return;
			}
			Coord ip = Coord(i, j, k);

			Coord alphav = fp - ip;
			Real alpha = alphav[0];
			Real beta = alphav[1];
			Real gamma = alphav[2];

			Real d000 = gDist(i, j, k);
			Real d100 = gDist(i + 1, j, k);
			Real d010 = gDist(i, j + 1, k);
			Real d110 = gDist(i + 1, j + 1, k);
			Real d001 = gDist(i, j, k + 1);
			Real d101 = gDist(i + 1, j, k + 1);
			Real d011 = gDist(i, j + 1, k + 1);
			Real d111 = gDist(i + 1, j + 1, k + 1);

			Real dx00 = Lerp(d000, d100, alpha);
			Real dx10 = Lerp(d010, d110, alpha);
			Real dxy0 = Lerp(dx00, dx10, beta);

			Real dx01 = Lerp(d001, d101, alpha);
			Real dx11 = Lerp(d011, d111, alpha);
			Real dxy1 = Lerp(dx01, dx11, beta);

			Real d0y0 = Lerp(d000, d010, beta);
			Real d0y1 = Lerp(d001, d011, beta);
			Real d0yz = Lerp(d0y0, d0y1, gamma);

			Real d1y0 = Lerp(d100, d110, beta);
			Real d1y1 = Lerp(d101, d111, beta);
			Real d1yz = Lerp(d1y0, d1y1, gamma);

			Real dx0z = Lerp(dx00, dx01, gamma);
			Real dx1z = Lerp(dx10, dx11, gamma);

			g[0] = d0yz - d1yz;
			g[1] = dx0z - dx1z;
			g[2] = dxy0 - dxy1;

			Real l = g.norm();
			if (l < 0.0001f) g = Coord(0);
			else g = g.normalize();

			d = (1.0f - gamma) * dxy0 + gamma * dxy1;
		}

		COMM_FUNC inline Real Lerp(Real a, Real b, Real alpha) const {
			return (1.0f-alpha)*a + alpha *b;
		}

	public:
		void ReadSDF(std::string filename);

	public:
		Coord left;		// lower left front corner
		Coord h;			// single cell sizes

		DeviceArray3D<Real> gDist;

		bool bInvert;
	};


	template class DistanceField3D<DataType3f>;
	template class DistanceField3D<DataType3d>;
}

#endif