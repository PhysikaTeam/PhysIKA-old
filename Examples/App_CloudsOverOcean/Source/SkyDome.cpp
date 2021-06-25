#include "Common.h"
#define NUM_SKYCOLORS 16

//a nice Norwegian sky, using a D-Image Z3 camera
// Oslo, Fagerborg, 17/8/2005, 12:41
unsigned char SkyColors[NUM_SKYCOLORS][3] = 
{ 
	{180, 222, 238},	
	{175, 218, 236},
	{155, 207, 232},
	{146, 201, 229},
	{139, 195, 225},
	{131, 188, 221},
	{127, 186, 220},
	{117, 177, 215},

	{104, 162, 202},
	{91, 148, 191},
	{87, 143, 187},
	{79, 135, 180},
	{79, 132, 177},
	{76, 128, 176},
	{71, 123, 170}, 
	{69, 119, 165}
};

CSkyDome::CSkyDome()
{
	SunPhi=SUN_AZIMUTH-M_PI/2; 
    SunTheta=(M_PI/2-SUN_ELEVATION);

}

int CSkyDome::Initialize(float Radius, int Slices, int Sides, bool exponential)
{
	NumSides = Sides;
	NumSlices = Slices;
	NumIndices = Slices * (Sides + 1) * 2;
	SkyRadius = Radius;

	VertexBuffer = new Vector3[(Slices + 1) * (Sides + 1)];
	ColorBuffer = new Color4[(Slices + 1) * (Sides + 1)];

	IndexBuffer = new unsigned short[NumIndices];

	float polyAng = 2.0f * PI / Sides, ang;

	float vx, vy, vz;

	int i, j;
	for (j = 0; j <= Slices; j++)
	{
		float val = (float)j / (float)Slices;
		ang = val * val * (PI / 2);
		for (i = 0; i <= Sides; i++)
		{
			vx = cos(i * polyAng) * cos(ang);
			vy = sin(ang);
			vz = sin(i * polyAng) * cos(ang);

			VertexBuffer[j * (Sides + 1) + i].x = vx * SkyRadius;
			VertexBuffer[j * (Sides + 1) + i].z = vz * SkyRadius;
			VertexBuffer[j * (Sides + 1) + i].y = vy * SkyRadius;

			//some magical color interpolation
			float fY = (float)j / (float)NumSlices * (float)NUM_SKYCOLORS;
			int index0 = fY;
			int index1 = (index0 > NUM_SKYCOLORS - 1 ? index1 = NUM_SKYCOLORS - 1 : index1 = index0 + 1);
			float f = fY - index0;

			Color4 c1(SkyColors[index0][0] / 255.0f, SkyColors[index0][1] / 255.0f, SkyColors[index0][2] / 255.0f, 1.0f);
			Color4 c2(SkyColors[index1][0] / 255.0f, SkyColors[index1][1] / 255.0f, SkyColors[index1][2] / 255.0f, 1.0f);

			ColorBuffer[j * (Sides + 1) + i] = c1 * (1.0f - f) + c2 * f;
		}
	}

	int ind = 0;
	for (j = 1; j <= Slices; j++)
	{
		for (i = 0; i <= Sides; i++)
		{
			IndexBuffer[ind++] = j * (Sides + 1) + i;
			IndexBuffer[ind++] = (j - 1) * (Sides + 1) + i;
		}
	}

	unsigned ImgID;
	ilGenImages(1, &ImgID);
	ilBindImage(ImgID);

	string tmp = get_project_source_dir() + "/Miscellaneous/flare0.bmp";
	if (ilLoadImage(ILstring(const_cast<char*>(tmp.c_str()))) == false)
	{
		cout << "Cannot load flare0.bmp" << endl;
		return 1;
	}

	glGenTextures(1, &FlareTex0);
	glBindTexture(GL_TEXTURE_2D, FlareTex0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ilGetInteger(IL_IMAGE_WIDTH),
		ilGetInteger(IL_IMAGE_HEIGHT), 0, GL_LUMINANCE,
		GL_UNSIGNED_BYTE, ilGetData());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	ilDeleteImages(1, &ImgID);

	return 0;
}

void CSkyDome::Destroy()
{
	delete [] VertexBuffer;
	delete [] IndexBuffer;
}

void CSkyDome::Render(Vector3 Camera)
{
	glPushMatrix();
	glTranslatef(Camera.x, Camera.y, Camera.z);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, &VertexBuffer[0]);
	glColorPointer(4, GL_FLOAT, 0, &ColorBuffer[0]);

	for (int i = 0; i < NumSlices; i++)
	{
		glDrawElements(GL_TRIANGLE_STRIP, (NumSides + 1) * 2, GL_UNSIGNED_SHORT, &IndexBuffer[i * (NumSides + 1) * 2]);
	}

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	Vector3 SunPos(sin(SunTheta) * cos(SunPhi) * SkyRadius,
		cos(SunTheta) * SkyRadius,
		sin(SunTheta) * sin(SunPhi) * SkyRadius);

	float SunSize = 20;
	Vector3 Vertices[4];

	float mat[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, mat);

	Vector3 vx(mat[0], mat[4], mat[8]);
	Vector3 vy(mat[1], mat[5], mat[9]);

	Vertices[0] = (SunPos + (vx + vy) * -SunSize * 3);
	Vertices[1] = (SunPos + (vx - vy) * SunSize * 3);
	Vertices[2] = (SunPos + (vx + vy) * SunSize * 3);
	Vertices[3] = (SunPos + (vy - vx) * SunSize * 3);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, FlareTex0);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);

	glColor4f(1.0f, 1.0f, 1.0f, 0.8f);

	glBegin(GL_QUADS);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3fv(!Vertices[0]);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3fv(!Vertices[1]);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3fv(!Vertices[2]);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3fv(!Vertices[3]);
	glEnd();

	glDisable(GL_BLEND);

	glPopMatrix();
}
