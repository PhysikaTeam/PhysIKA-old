#ifndef _COLOR_H
#define _COLOR_H
#include <math.h>
#define F_ZERO (1.0 / 999999999)
class Color3
{
public:

	Color3()
	{
		R = G = B = 0.0f;
	}

	Color3(float cR, float cG, float cB)
	{
		if (cR > 1.0f) cR = 1.0f; if (cG > 1.0f) cG = 1.0f;	if (cB > 1.0f) cB = 1.0f;
		if (cR < 0.0f) cR = 0.0f; if (cG < 0.0f) cG = 0.0f;	if (cB < 0.0f) cB = 0.0f;

		R = cR;
		G = cG;
		B = cB;
	}

	float* operator!()
	{
		return (float*) this;
	}


	Color3& operator+=(Color3 other)
	{
		R += other.R;
		G += other.G;
		B += other.B;

		return  *this;
	}


	Color3 operator+(Color3 other)
	{
		return  Color3(R + other.R, G + other.G, B + other.B);

	}

	Color3 operator-(Color3 other)
	{
		return  Color3(R - other.R, G - other.G, B - other.B);

	}

	Color3 operator*(float num)
	{
		return Color3(R* num, G * num, B * num);
	}

	void Clamp()
	{
		if (R > 1.0f) R = 1.0f;	if (G > 1.0f) G = 1.0f;	if (B > 1.0f) B = 1.0f;
		if (R < 0.0f) R = 0.0f;	if (G < 0.0f) G = 0.0f;	if (B < 0.0f) B = 0.0f;
	}
	float maxPart()
	{
		float tempMax = -99999;
		if (tempMax<R)
			tempMax = R;
		if (tempMax<G)
			tempMax = G;
		if (tempMax<B)
			tempMax = B;

		return tempMax;
	}

	float  minPart()
	{
		float tempMin = 99999;
		if (tempMin>R)
			tempMin = R;
		if (tempMin>G)
			tempMin = G;
		if (tempMin>B)
			tempMin = B;
		return tempMin;
	}

	Color3  RGB2xyY()
	{
		// http://zh.wikipedia.org/wiki/SRGB%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4

		float rgb[3] = { R,G,B };


		float CM[3][3] =
		{
			{ 0.4124f, 0.3576f, 0.1805f },
			{ 0.2126f, 0.7152f, 0.0722f },
			{ 0.0193f, 0.1192f, 0.9505f }
		};
		float XYZ[3];


		XYZ[0] = (CM[0][0] * rgb[0] + CM[0][1] * rgb[1] + CM[0][2] * rgb[2]);
		XYZ[1] = (CM[1][0] * rgb[0] + CM[1][1] * rgb[1] + CM[1][2] * rgb[2]);
		XYZ[2] = (CM[2][0] * rgb[0] + CM[2][1] * rgb[1] + CM[2][2] * rgb[2]);

		// XYZ -> xyY conversion		
		Color3 xyY;
		xyY.B = XYZ[1];

		float inv_sum = XYZ[0] + XYZ[1] + XYZ[2];
		if (inv_sum<0.001)
		{
			inv_sum = 0.001;
		}
		xyY.R = XYZ[0] / inv_sum;

		xyY.G = XYZ[1] / inv_sum;

		return xyY;

	}


	float g(float x)
	{
		float a = 0.055;
		if (x>0.04045)
			return powf((x + a) / (1 + a), 2.4);
		else
			return x / 12.92;

	}


	Color3 xyY2RGB()
	{
		float xyY[3] = { R,G,B };

		//first convert to XYZ
		float XYZ[3];

		if (fabs(xyY[1]) < F_ZERO)
		{
			XYZ[0] = XYZ[1] = XYZ[2] = 0;
		}
		else
		{
			XYZ[1] = xyY[2];
			XYZ[0] = xyY[0] * xyY[2] / xyY[1];
			XYZ[2] = (1.0f - xyY[0] - xyY[1]) * xyY[2] / xyY[1];
		}


		//for XYZ2rgb
		float CM[3][3] =
		{
			{ 3.2410f, -1.5374f, -0.4986f },
			{ -0.9692f, 1.8760f, 0.0416f },
			{ 0.0556f, -0.2040f, 1.0570f }

		};


		float rgb[3];
		rgb[0] = (CM[0][0] * XYZ[0] + CM[0][1] * XYZ[1] + CM[0][2] * XYZ[2]);
		rgb[1] = (CM[1][0] * XYZ[0] + CM[1][1] * XYZ[1] + CM[1][2] * XYZ[2]);
		rgb[2] = (CM[2][0] * XYZ[0] + CM[2][1] * XYZ[1] + CM[2][2] * XYZ[2]);


		for (int i = 0; i<3; i++)
		{
			if (rgb[i]<0.00304)
			{
				rgb[i] *= 12.92;
				if (rgb[i]<0)
					rgb[i] = 0;
			}
			else
			{
				rgb[i] = (1 + 0.055)*powf(rgb[i], 1.0 / 2.4) - 0.055;
				if (rgb[i]>1)
					rgb[i] = 1;
			}
		}

		return Color3(rgb[0], rgb[1], rgb[2]);
	}

public:
	float R, G, B;

};




inline float Dot(Color3 v1, Color3 v2)
{
	return (float)(v1.R* v2.R + v1.G * v2.G + v1.B * v2.B);
}

class Color4
{
public:

	Color4()
	{
		R = G = B = A = 0.0f;
	}

	Color4(float cR, float cG, float cB, float cA)
	{
		if (cR > 1.0f) cR = 1.0f;	if (cG > 1.0f) cG = 1.0f;
		if (cB > 1.0f) cB = 1.0f;	if (cA > 1.0f) cA = 1.0f;
		if (cR < 0.0f) cR = 0.0f;	if (cG < 0.0f) cG = 0.0f;
		if (cB < 0.0f) cB = 0.0f;	if (cA < 0.0f) cA = 0.0f;

		R = cR;		G = cG;		B = cB;		A = cA;
	}

	float* operator!()
	{
		return (float*) this;
	}

	Color4 operator+(Color4 c)
	{
		Color4 res;

		res.R = R + c.R;		res.G = G + c.G;
		res.B = B + c.B;		res.A = A + c.A;

		return res;
	}

	Color4 operator-(Color4 c)
	{
		Color4 res;
		res.R = R - c.R;		res.G = G - c.G;
		res.B = B - c.B;		res.A = A - c.A;
		return res;
	}

	Color4 operator*(float f)
	{
		Color4 res;
		res.R = R * f;		res.G = G * f;
		res.B = B * f;		res.A = A * f;
		return res;
	}

	void Clamp()
	{
		if (R > 1.0f) R = 1.0f;		if (G > 1.0f) G = 1.0f;
		if (B > 1.0f) B = 1.0f;		if (A > 1.0f) A = 1.0f;
		if (R < 0.0f) R = 0.0f;		if (G < 0.0f) G = 0.0f;
		if (B < 0.0f) B = 0.0f;		if (A < 0.0f) A = 0.0f;
	}

	float R, G, B, A;
};

#endif
