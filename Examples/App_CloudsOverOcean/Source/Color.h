#ifndef _COLOR_H
#define _COLOR_H

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

	void Clamp()
	{
		if (R > 1.0f) R = 1.0f;	if (G > 1.0f) G = 1.0f;	if (B > 1.0f) B = 1.0f;
		if (R < 0.0f) R = 0.0f;	if (G < 0.0f) G = 0.0f;	if (B < 0.0f) B = 0.0f;
	}

	float R, G, B;
};

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



