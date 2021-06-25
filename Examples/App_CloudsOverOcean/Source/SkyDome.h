
#ifndef _SKY_DOME_H
#define _SKY_DOME_H

class CSkyDome
{
	public:
		CSkyDome();
	
		int  Initialize(float Radius, int NumSlices, int NumSides, bool exponential);
		void Render(Vector3 Camera);
		void Destroy();

		Vector3 GetSunVector()
		{
			return Vector3(sin(SunTheta) * cos(SunPhi) * SkyRadius,
			    			cos(SunTheta) * SkyRadius,
			    			sin(SunTheta) * sin(SunPhi) * SkyRadius);
		}
		
	
public:
			
		Vector3 *VertexBuffer;
		Color4*	ColorBuffer;
	
		unsigned short *IndexBuffer;

		int NumSlices;
		int NumSides;
		int NumIndices;
		
		float SkyRadius;
		
		unsigned	FlareTex0, FlareTex1;

		float SunTheta, SunPhi;
		
		float TimeOfDay, JulianDay, Latitude, Longitude;		
};

#endif
