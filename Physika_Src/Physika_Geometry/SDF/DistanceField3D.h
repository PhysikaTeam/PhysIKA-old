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
#include "Physika_Core/Cuda_Array/Array3D.h"

namespace Physika {

	class DistanceField3D {
	public:
		DistanceField3D();

		DistanceField3D(std::string filename);

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~DistanceField3D();

		void SetSpace(const float3 p0, const float3 p1, int nbx, int nby, int nbz);
		void Release();

		void Translate(const float3& t);
		void Scale(const float s);
		void Invert();

		void DistanceFieldToBox(float3& lo, float3& hi, bool inverted);

		void DistanceFieldToCylinder(float3& center, float radius, float height, int axis, bool inverted);

		void DistanceFieldToSphere(float3& center, float radius, bool inverted);

		HYBRID_FUNC void GetDistance(const float3 &p, float &d, float3 &g) {
			// get cell and lerp values
			float3 fp = (p - left)*make_float3(1.0f/h.x, 1.0f/h.y, 1.0f/h.z);
			const int i = (int)floor(fp.x);
			const int j = (int)floor(fp.y);
			const int k = (int)floor(fp.z);
			if (i < 0 || i >= gDist.Nx() - 1 || j < 0 || j >= gDist.Ny() - 1 || k < 0 || k >= gDist.Nz()-1) {
				if (bInvert) d = -100000.0f;
				else d = 100000.0f;
				g = make_float3(0.0f);
				return;
			}
			float3 ip = make_float3(i, j, k);

			float3 alphav = fp - ip;
			float alpha = alphav.x;
			float beta = alphav.y;
			float gamma = alphav.z;

			float d000 = gDist(i, j, k);
			float d100 = gDist(i + 1, j, k);
			float d010 = gDist(i, j + 1, k);
			float d110 = gDist(i + 1, j + 1, k);
			float d001 = gDist(i, j, k + 1);
			float d101 = gDist(i + 1, j, k + 1);
			float d011 = gDist(i, j + 1, k + 1);
			float d111 = gDist(i + 1, j + 1, k + 1);

			float dx00 = Lerp(d000, d100, alpha);
			float dx10 = Lerp(d010, d110, alpha);
			float dxy0 = Lerp(dx00, dx10, beta);

			float dx01 = Lerp(d001, d101, alpha);
			float dx11 = Lerp(d011, d111, alpha);
			float dxy1 = Lerp(dx01, dx11, beta);

			float d0y0 = Lerp(d000, d010, beta);
			float d0y1 = Lerp(d001, d011, beta);
			float d0yz = Lerp(d0y0, d0y1, gamma);

			float d1y0 = Lerp(d100, d110, beta);
			float d1y1 = Lerp(d101, d111, beta);
			float d1yz = Lerp(d1y0, d1y1, gamma);

			float dx0z = Lerp(dx00, dx01, gamma);
			float dx1z = Lerp(dx10, dx11, gamma);

			g.x = d0yz - d1yz;
			g.y = dx0z - dx1z;
			g.z = dxy0 - dxy1;

			float l = length(g);
			if (l < 0.0001f) g = make_float3(0.0f);
			else g = normalize(g);

			d = (1.0f - gamma) * dxy0 + gamma * dxy1;
		}

		HYBRID_FUNC inline float Lerp(float a, float b, float alpha) const {
			return (1.0f-alpha)*a + alpha *b;
		}

	public:
		void ReadSDF(std::string filename);

	public:
		float3 left;		// lower left front corner
		float3 h;			// single cell sizes

		Grid1f gDist;

		bool bInvert;
	};

}

#endif