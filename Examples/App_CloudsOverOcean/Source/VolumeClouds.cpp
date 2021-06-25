#include "Common.h"

#define GL_CLAMP_TO_EDGE 0x812F
#define DIST_MUL_IMPOSTOR 	4//used for impostor/3d distance interpolation
#define DIST_MUL_3D			2		//same as above
#define ALPHA_ADJUST		0.3f	//tweak the fading between impostors and 3d models so we 
//don't get transparent clouds

float min_ext=MAXVAL;
float max_ext=-MAXVAL;

extern  CFrustum 			Frustum;

extern GLuint g_framebuffer;
extern GLuint g_texture;
extern GLuint g_depthbuffer;

VolumetricClouds::VolumetricClouds()
{
	Albedo = 1.0f;
	Extinction =85.0f;
	SplatBufferSize = 32;		//size of the splat buffer (this is a good value)
	ImpostorSize = 512;			//maximum size of the impostor
}

void VolumetricClouds::GrowCloud(int cloud_id, VolumetricCloud* Cloud, int level, float radius, Vector3 Position)
{
	FILE* fp = NULL;
	char strname[256];
	//sprintf(strname, "input\\particle\\cloud%d.dat", cloud_id);
	sprintf(strname, "/Input/particle_example/cloud%d.dat", cloud_id);
	string fullname = get_project_source_dir() + string(strname);

	fp = fopen(fullname.c_str(), "rb");
	if (!fp) return;

	int Num;
	fread(&Num, sizeof(int), 1, fp);

	CloudPuff	puff;
	//compute the bounding box, used for rendering impostors and frustum culling
	Cloud->BoundingBox1 = Vector3(-MAXVAL, -MAXVAL, -MAXVAL);
	Cloud->BoundingBox2 = Vector3(MAXVAL, MAXVAL, MAXVAL);
	for (int i = 0; i < Num; i++)
	{
		fread(&puff.Position, sizeof(Vector3), 1, fp);
		puff.Position = puff.Position * scale_particle;

		if (puff.Position.x > Cloud->BoundingBox1.x) Cloud->BoundingBox1.x = puff.Position.x;
		if (puff.Position.y > Cloud->BoundingBox1.y) Cloud->BoundingBox1.y = puff.Position.y;
		if (puff.Position.z > Cloud->BoundingBox1.z) Cloud->BoundingBox1.z = puff.Position.z;
		if (puff.Position.x < Cloud->BoundingBox2.x) Cloud->BoundingBox2.x = puff.Position.x;
		if (puff.Position.y < Cloud->BoundingBox2.y) Cloud->BoundingBox2.y = puff.Position.y;
		if (puff.Position.z < Cloud->BoundingBox2.z) Cloud->BoundingBox2.z = puff.Position.z;

		puff.Position += Position;
		puff.Angle = 0.0f;
		puff.ID = i;
		puff.Life = 1.0f;

		Cloud->Puffs.push_back(puff);
	}


	Cloud->Center = (Cloud->BoundingBox1 + Cloud->BoundingBox2) * 0.5 + Position;
	Cloud->Radius = Magnitude(Cloud->BoundingBox1 - Cloud->BoundingBox2) * 0.5;

	for (int i = 0; i < Num; i++)
	{
		fread(&Cloud->Puffs[i].Size, sizeof(float), 1, fp);
		Cloud->Puffs[i].Size *= scale_particle;
	}
	for (int i = 0; i < Num; i++)
	{
		fread(&Cloud->Puffs[i].Color, sizeof(Color4), 1, fp);
	}
	for (int i = 0; i < Num; i++)
	{
		fread(&Cloud->Puffs[i].Extinction, sizeof(float), 1, fp);

		max_ext = max(max_ext, Cloud->Puffs[i].Extinction);
		min_ext = min(min_ext, Cloud->Puffs[i].Extinction);

	}

	fclose(fp);

	//allocate buffers for rendering
	Cloud->VertexBuffer = new Vector3[Num * 4];
	Cloud->TexCoordBuffer = new Vector2[Num * 4];
	Cloud->ColorBuffer = new Color4[Num * 4];
}

void VolumetricClouds::GenerateTexture()
{
	int N = 64;
	unsigned char* B = new unsigned char[4 * N * N];
	float X, Y, Dist;
	float Incr = 2.0f / N;
	int i = 0, j = 0;
	float value;

	Y = -1.0f;
	for (int y = 0; y < N; y++)
	{
		X = -1.0f;
		for (int x = 0; x < N; x++, i++, j += 4)
		{
			Dist = (float)sqrt(X * X + Y * Y);
			if (Dist > 1) Dist = 1;

			//our magical interpolation polynomical
			value = 2 * Dist * Dist * Dist - 3 * Dist * Dist + 1;
			value *= 0.4f;

			B[j + 3] = B[j + 2] = B[j + 1] = B[j] = (unsigned char)(value * 255);

			X += Incr;
		}
		Y += Incr;
	}

	glGenTextures(1, &PuffTexture);
	glBindTexture(GL_TEXTURE_2D, PuffTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, N, N, 0, GL_RGBA, GL_UNSIGNED_BYTE, B);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	delete[] B;
}

int VolumetricClouds::Create(int start, int Num, Vector3 RelativePos)
{
	int i;
	effective_number = 0;

	for (i = start; i < start + Num; i++)
	{
		VolumetricCloud Cloud;
		Cloud.Center.x = 0;
		Cloud.Center.y = 0;
		Cloud.Center.z = 0;
		Cloud.Radius = 15.0f;
		Cloud.LastLight = Vector3(0, 0, 0);
		Cloud.LastCamera = Vector3(0, 0, 0);
		Cloud.ImpostorTex = 0;

		GrowCloud(i, &Cloud, 0, Cloud.Radius, RelativePos);

		if (Cloud.Puffs.size() > 0)
		{
			Clouds.push_back(Cloud);
			effective_number++;
		}
	}
	GenerateTexture();
	NormalizeExtinction();

	return 0;
}

void VolumetricClouds::Update(Vector3 Sun, Vector3 Camera)
{
	Vector3 SunDir, ToCam;

	SunDir = Normalize(Sun);

	for (unsigned i = 0; i < Clouds.size(); i++)
	{
		//if the angle between the camera and the cloud center
		//has changed enough since the last lighting calculation
		//recalculate the lighting (very slow - don't move the sun :) )

		ToCam = Normalize(Camera - Clouds[i].Center);
		if (Dot(SunDir, Clouds[i].LastLight) < 0.99f)
		{
			LightCloud(&Clouds[i], Sun);
			MakeCloudImpostor(&Clouds[i], Sun, Camera);
			Clouds[i].LastLight = SunDir;
			Clouds[i].LastCamera = ToCam;
		}
		else
		{
			float dot = Dot(ToCam, Clouds[i].LastCamera);
			bool in_frustum = Frustum.SphereInFrustum(Clouds[i].Center, Clouds[i].Radius);
			int mip_size = GetImpostorSize(SqDist(Camera, Clouds[i].Center));

			//same as above only recreating the impostor
			//also recreates the impostor if the current mip level
			//isn't good enough (the camera has come too close)
			if ((dot < 0.99f || Clouds[i].ImpostorSize < mip_size) && in_frustum)
			{
				MakeCloudImpostor(&Clouds[i], Sun, Camera);
				Clouds[i].LastCamera = ToCam;
			}
		}
	}
}

int VolumetricClouds::GetImpostorSize(float d)
{
	//this is done rather arbitrarely
	//it all depends on the scale of the world/clouds
	//to make it more robust we should use the size of
	//the cloud
	int Size = ImpostorSize;

	if (d > 50 * 50) Size /= 2;
	if (d > 100 * 100) Size /= 2;
	if (d > 200 * 200) Size /= 2;
	if (d > 300 * 300) Size /= 2;

	if (Size < 32) Size = 32;

	return Size;
}

void VolumetricClouds::Render(Vector3 Camera, Vector3 Sun)
{
	unsigned i, j;

	NumSprites = NumImpostors = 0;

	for (i = 0; i < Clouds.size(); i++)
	{
		Clouds[i].DistanceFromCamera = SqDist(Clouds[i].Center, Camera);
	}
	//we sort our clouds (impostors) when doing rendering since we're alpha blending
	bool done = false;
	while (!done)
	{
		done = true;
		for (i = 0; i < Clouds.size(); i++)
			for (j = i + 1; j < Clouds.size(); j++)
			{
				if (Clouds[i].DistanceFromCamera < Clouds[j].DistanceFromCamera)
				{
					swap(Clouds[i], Clouds[j]);
					done = false;
				}
			}
	}

	float dist_impostor, dist_3D;
	float alpha;

	for (i = 0; i < Clouds.size(); i++)
	{
		dist_impostor = Clouds[i].Radius * DIST_MUL_IMPOSTOR; //beyond this render only the impostor
		dist_impostor *= dist_impostor;	//square this since the camera distance is also squared
		dist_3D = Clouds[i].Radius * DIST_MUL_3D;//closer than this render only the 3D model
		dist_3D *= dist_3D; //square ourselves

		//if we're in between we need to interpolate

		if (!Frustum.SphereInFrustum(Clouds[i].Center, Clouds[i].Radius)) continue;

		RenderCloud3D(&Clouds[i], Camera, Sun, 1.0f);
		continue;


		if (Clouds[i].DistanceFromCamera > dist_impostor)
			//fully impostor
			RenderCloudImpostor(&Clouds[i], 1.0f);
		else
		{
			if (Clouds[i].DistanceFromCamera < dist_3D)
				//fully 3D
				RenderCloud3D(&Clouds[i], Camera, Sun, 1.0f);
			else
			{
				//in between, interpolate nicely and tweak the alpha to make it look prettier
				alpha = (Clouds[i].DistanceFromCamera - dist_3D) / (dist_impostor - dist_3D);
				RenderCloudImpostor(&Clouds[i], ALPHA_ADJUST + alpha);
				RenderCloud3D(&Clouds[i], Camera, Sun, 1.0f - alpha);
			}
		}
	}
}

void VolumetricClouds::MakeCloudImpostor(VolumetricCloud* Cloud, Vector3 Sun, Vector3 Camera)
{
	//2014/4/26,YCQ
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, g_framebuffer);
	//2014/4/26,YCQ

	unsigned i;
	//puffs are in world space
	for (i = 0; i < Cloud->Puffs.size(); i++)
		Cloud->Puffs[i].DistanceToCam = SqDist(Camera, Cloud->Puffs[i].Position);

	SortParticles(Cloud, SORT_TOWARD);

	int 	ViewportSize = ImpostorSize;

	float d = Dist(Camera, Cloud->Center);
	float r = Cloud->Radius;
	float pr = Cloud->Puffs[0].Size;

	ViewportSize = GetImpostorSize(d * d);

	//if it's the first time we render this, use full resolution to create a big
	//texture. Otherwise, when we get a bigger impostor size than the original one,
	//glCopyTexSubImage2D will fail
	if (glIsTexture(Cloud->ImpostorTex) == GL_FALSE)
		ViewportSize = ImpostorSize;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-Cloud->Radius - pr, Cloud->Radius + pr, -Cloud->Radius - pr, Cloud->Radius + pr, d - r, d + r);

	//we setup the camera to look at the cloud center from the camera position
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	gluLookAt(Camera.x, Camera.y, Camera.z, Cloud->Center.x, Cloud->Center.y, Cloud->Center.z, 0, 1, 0);

	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0, 0, ViewportSize, ViewportSize);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	float mat[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, mat);

	Vector3 vx(mat[0], mat[4], mat[8]);
	Vector3 vy(mat[1], mat[5], mat[9]);

	Cloud->vx = vx;		Cloud->vy = vy; //store for rendering

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, PuffTexture);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	Vector3 Light = Normalize(Camera - Sun);
	Vector3 Omega;
	Color4  ParticleColor;

	vector<CloudPuff>::iterator PuffIter;
	CloudPuff* Puff;
	int index = 0;

	Vector3* vp; Vector2* tp; Color4* cp;
	vp = Cloud->VertexBuffer;
	tp = Cloud->TexCoordBuffer;
	cp = Cloud->ColorBuffer;

	Vector2 v1(1.0f, 0.0f), v2(0.0f, 0.0f), v3(0.0f, 1.0f), v4(1.0f, 1.0f);
	float costheta, phase;
	float omega2;

	Vector3 Corner1, Corner2, Corner3, Corner4;
	Corner1 = -vx - vy; Corner2 = vx - vy; Corner3 = vx + vy; Corner4 = vy - vx;

	for (PuffIter = Cloud->Puffs.begin(); PuffIter != Cloud->Puffs.end(); ++PuffIter)
	{
		Puff = &*PuffIter;
		Omega = Puff->Position - Camera;

		//we want to compute the phase function
		omega2 = Omega.x * Omega.x + Omega.y * Omega.y + Omega.z * Omega.z;
		omega2 = carmack_func(omega2);

		//omega2 is now 1 / Mag(Omega)
		Omega.x *= omega2;	Omega.y *= omega2;	Omega.z *= omega2;

		//and this is the phase function
		costheta = Dot(Omega, Light);
		phase = PhaseFunction(costheta); //0.75f * (1.0f + costheta * costheta);

		ParticleColor.R = AMBIENT_R + Puff->Color.R * phase;
		ParticleColor.G = AMBIENT_G + Puff->Color.G * phase;
		ParticleColor.B = AMBIENT_B + Puff->Color.B * phase;
		ParticleColor.A = Puff->Color.A * Puff->Life;

		//add everything to the buffers
		*(vp++) = Puff->Position + Corner1 * Puff->Size;
		*(tp++) = v1;
		*(cp++) = ParticleColor;

		*(vp++) = Puff->Position + Corner2 * Puff->Size;
		*(tp++) = v2;
		*(cp++) = ParticleColor;

		*(vp++) = Puff->Position + Corner3 * Puff->Size;
		*(tp++) = v3;
		*(cp++) = ParticleColor;

		*(vp++) = Puff->Position + Corner4 * Puff->Size;
		*(tp++) = v4;
		*(cp++) = ParticleColor;
	}

	//render using vertex arrays
	//homework: 1. try using dynamic vertex buffers and see the difference
	//(it probably won't be much since we're fillrate limited
	//homework: 2. also try using triangles instead of quads to see the difference
	//(i guess that depends on the card)

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, Cloud->VertexBuffer);
	glColorPointer(4, GL_FLOAT, 0, Cloud->ColorBuffer);
	glTexCoordPointer(2, GL_FLOAT, 0, Cloud->TexCoordBuffer);

	glDrawArrays(GL_QUADS, 0, Cloud->Puffs.size() * 4);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	glDisable(GL_BLEND);
	glDisable(GL_TEXTURE_2D);

	glPopAttrib();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	//if we haven't yet created an impostor texture, well, create one
	//anyway, copy from the framebuffer to the impostor texture
	//glCopyTexSubImage2D is faster so we use that if we're reloading the impostor
	if (glIsTexture(Cloud->ImpostorTex) == GL_FALSE)
	{
		glGenTextures(1, &Cloud->ImpostorTex);
		glBindTexture(GL_TEXTURE_2D, Cloud->ImpostorTex);
		glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 0, 0, ViewportSize, ViewportSize, 0);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}
	else
	{
		glBindTexture(GL_TEXTURE_2D, Cloud->ImpostorTex);
		glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, ViewportSize, ViewportSize);
	}

	Cloud->ImpostorSize = ViewportSize;

	//2014/4/26,YCQ
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	unsigned e;
	if ((e = glGetError()))
	{
		cout << "OpenGL Error: " << gluErrorString(e) << endl;
		//return 1;
	}
	//2014/4/26,YCQ
}

void VolumetricClouds::LightCloud(VolumetricCloud* Cloud, Vector3 Sun)
{
	//2014/4/26,YCQ
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, g_framebuffer);
	//2014/4/26,YCQ

	float  Distance2Sun_Scale = 1000;
	//	assume Sun is a point!!
	Sun = Normalize(Sun) * Cloud->Radius * 1.5f * Distance2Sun_Scale + Cloud->Center;

	//	puffs are now in world space
	unsigned i, j;
	for (i = 0; i < Cloud->Puffs.size(); i++)
		Cloud->Puffs[i].DistanceToCam = SqDist(Sun, Cloud->Puffs[i].Position);

	SortParticles(Cloud, SORT_AWAY);

	float d = Dist(Sun, Cloud->Center);
	float r = Cloud->Radius;
	float pr = Cloud->Puffs[0].Size;

	//we setup an orthographic projection
	//the view volume will thus be a box and it will fit the cloud perfectly
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-Cloud->Radius - pr, Cloud->Radius + pr, -Cloud->Radius - pr, Cloud->Radius + pr, d - r, d + r);

	//setup the camera to lookat the cloud center and be positioned on the sun
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	gluLookAt(Sun.x, Sun.y, Sun.z, Cloud->Center.x, Cloud->Center.y, Cloud->Center.z, 0, 1, 0);

	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0, 0, SplatBufferSize, SplatBufferSize);

	//clear our buffer make it fully bright
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//your standard up and right vector extraction
	float mat[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, mat);

	Vector3 vx(mat[0], mat[4], mat[8]);
	Vector3 vy(mat[1], mat[5], mat[9]);


	float SolidAngle = 0.09f, Area;
	unsigned Pixels;
	double mp[16], mm[16];
	int vp[4];
	float* buf, avg;

	//used for projection
	glGetDoublev(GL_MODELVIEW_MATRIX, mm);
	glGetDoublev(GL_PROJECTION_MATRIX, mp);
	glGetIntegerv(GL_VIEWPORT, vp);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, PuffTexture);

	//our blending is enabled
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_GREATER, 0);

	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);


	Color4 LightColor = SUN_COLOR;

	Color4 ParticleColor;
	float factor = 1.0f;
	int ReadX, ReadY;

	Vector3 Light = Normalize(Sun);

	factor = SolidAngle / (4 * PI);

	for (i = 0; i < Cloud->Puffs.size(); i++)
	{
		//find the particle's projected coordinates
		double CenterX, CenterY, CenterZ;
		gluProject(Cloud->Puffs[i].Position.x,
			Cloud->Puffs[i].Position.y,
			Cloud->Puffs[i].Position.z,
			mm, mp, vp, &CenterX, &CenterY, &CenterZ);

		//area = dist * dist * angle
		Area = Cloud->Puffs[i].DistanceToCam / (Distance2Sun_Scale * Distance2Sun_Scale) * SolidAngle;
		Pixels = (int)(sqrt(Area) * SplatBufferSize / (2 * Cloud->Radius));
		if (Pixels < 1) Pixels = 1;

		//make sure we're not reading from outside the viewport, that's Undefined
		ReadX = (int)(CenterX - Pixels / 2);
		if (ReadX < 0) ReadX = 0;
		ReadY = (int)(CenterY - Pixels / 2);
		if (ReadY < 0) ReadY = 0;

		buf = new float[Pixels * Pixels];
		//we only need the red component since this is greyscale
		//glReadBuffer(GL_BACK);
		glReadPixels(ReadX, ReadY, Pixels, Pixels, GL_RED, GL_FLOAT, buf);

		avg = 0.0f;
		for (j = 0; j < Pixels * Pixels; j++) avg += buf[j];
		avg /= (Pixels * Pixels);

		delete[] buf;

		//Light color * 
		// average color from solid angle (sum * solidangle / (pixels^2 * 4pi)) 
		// * albedo * extinction 
		// * rayleigh scattering in the direction of the sun (1.5f) (only for rendering, don't store)
		float Extinction = Cloud->Puffs[i].Extinction;
		ParticleColor.R = SUN_INTENSITY * LightColor.R * Albedo * Extinction * avg * factor;
		ParticleColor.G = SUN_INTENSITY * LightColor.G * Albedo * Extinction * avg * factor;
		ParticleColor.B = SUN_INTENSITY * LightColor.B * Albedo * Extinction * avg * factor;
		ParticleColor.A = 1.0f - exp(-Extinction);

		Cloud->Puffs[i].Color = ParticleColor;
		Cloud->Puffs[i].Color.Clamp();

		//the phase function (it's always 1.5f when we're looking from the sun)
		ParticleColor = ParticleColor * 1.5f;
		ParticleColor.Clamp();

		glColor4fv(!ParticleColor);

		glBegin(GL_QUADS);

		glTexCoord2f(1.0f, 0.0f);
		glVertex3fv(!(Cloud->Puffs[i].Position + (vx + vy) * -Cloud->Puffs[i].Size));
		glTexCoord2f(0.0f, 0.0f);
		glVertex3fv(!(Cloud->Puffs[i].Position + (vx - vy) * Cloud->Puffs[i].Size));
		glTexCoord2f(0.0f, 1.0f);
		glVertex3fv(!(Cloud->Puffs[i].Position + (vx + vy) * Cloud->Puffs[i].Size));
		glTexCoord2f(1.0f, 1.0f);
		glVertex3fv(!(Cloud->Puffs[i].Position + (vy - vx) * Cloud->Puffs[i].Size));

		glEnd();
	}

	glDisable(GL_ALPHA_TEST);
	glDisable(GL_BLEND);
	glDisable(GL_TEXTURE_2D);

	glPopAttrib();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	//2014/4/26,YCQ
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	unsigned e;
	if ((e = glGetError()))
	{
		cout << "OpenGL Error: " << gluErrorString(e) << endl;
		//return 1;
	}

	//2014/4/26,YCQ
}

void VolumetricClouds::RenderCloud3D(VolumetricCloud* Cloud, Vector3 Camera, Vector3 Sun, float alpha)
{
	unsigned i;

	//puffs are in world space
	for (i = 0; i < Cloud->Puffs.size(); i++)
		Cloud->Puffs[i].DistanceToCam = SqDist(Camera, Cloud->Puffs[i].Position);

	SortParticles(Cloud, SORT_TOWARD);

	float mat[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, mat);

	Vector3 vx(mat[0], mat[4], mat[8]);
	Vector3 vy(mat[1], mat[5], mat[9]);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, PuffTexture);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	Vector3 Light = Normalize(Camera - Sun);
	Vector3 Omega;
	Color4  ParticleColor;

	vector<CloudPuff>::iterator PuffIter;
	CloudPuff* Puff;
	int index = 0;

	Vector3* vp; Vector2* tp; Color4* cp;
	vp = Cloud->VertexBuffer;
	tp = Cloud->TexCoordBuffer;
	cp = Cloud->ColorBuffer;

	Vector2 v1(1.0f, 0.0f), v2(0.0f, 0.0f), v3(0.0f, 1.0f), v4(1.0f, 1.0f);
	float costheta, phase;
	float omega2;

	Vector3 Corner1, Corner2, Corner3, Corner4;
	Corner1 = -vx - vy; Corner2 = vx - vy; Corner3 = vx + vy; Corner4 = vy - vx;

	for (PuffIter = Cloud->Puffs.begin(); PuffIter != Cloud->Puffs.end(); ++PuffIter)
	{
		Puff = &*PuffIter;
		Omega = Puff->Position - Camera;

		omega2 = Omega.x * Omega.x + Omega.y * Omega.y + Omega.z * Omega.z;
		omega2 = carmack_func(omega2);

		//omega2 is now 1 / Mag(Omega)
		Omega.x *= omega2;	Omega.y *= omega2;	Omega.z *= omega2;

		costheta = Dot(Omega, Light);
		phase = PhaseFunction(costheta); // 0.75f * (1.0f + costheta * costheta);

		ParticleColor.R = (AMBIENT_R + Puff->Color.R * phase) * alpha;
		ParticleColor.G = (AMBIENT_G + Puff->Color.G * phase) * alpha;
		ParticleColor.B = (AMBIENT_B + Puff->Color.B * phase) * alpha;

		ParticleColor.A = Puff->Color.A * Puff->Life * alpha;

		*(vp++) = Puff->Position + Corner1 * Puff->Size;
		*(tp++) = v1;
		*(cp++) = ParticleColor;

		*(vp++) = Puff->Position + Corner2 * Puff->Size;
		*(tp++) = v2;
		*(cp++) = ParticleColor;

		*(vp++) = Puff->Position + Corner3 * Puff->Size;
		*(tp++) = v3;
		*(cp++) = ParticleColor;

		*(vp++) = Puff->Position + Corner4 * Puff->Size;
		*(tp++) = v4;
		*(cp++) = ParticleColor;
	}

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, Cloud->VertexBuffer);
	glColorPointer(4, GL_FLOAT, 0, Cloud->ColorBuffer);
	glTexCoordPointer(2, GL_FLOAT, 0, Cloud->TexCoordBuffer);

	glDrawArrays(GL_QUADS, 0, Cloud->Puffs.size() * 4);
	NumSprites += Cloud->Puffs.size();

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	glDisable(GL_BLEND);
	glDisable(GL_TEXTURE_2D);
}

void VolumetricClouds::RenderCloudImpostor(VolumetricCloud* Cloud, float alpha)
{
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, Cloud->ImpostorTex);

	glEnable(GL_BLEND);
	glEnable(GL_ALPHA_TEST);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA); //_MINUS_SRC_ALPHA);
	glAlphaFunc(GL_GREATER, 0.0f);

	//we might get some hard edges because we don't clear the whole texture when
	//redrawing the impostor so a small adjustment of tex coords will fix this
	float texcoord = (float)(Cloud->ImpostorSize) / (float)ImpostorSize - 0.001f;

	//and here we make the cloud transparent depending on the alpha value
	//we use alpha in all components because of our special blending parameters
	glColor4f(alpha, alpha, alpha, alpha);

	glBegin(GL_QUADS);

	glTexCoord2f(0.0f, 0.0f);
	glVertex3fv(!(Cloud->Center + (Cloud->vx + Cloud->vy) * -Cloud->Radius));

	glTexCoord2f(texcoord, 0.0f);
	glVertex3fv(!(Cloud->Center + (Cloud->vx - Cloud->vy) * Cloud->Radius));

	glTexCoord2f(texcoord, texcoord);
	glVertex3fv(!(Cloud->Center + (Cloud->vx + Cloud->vy) * Cloud->Radius));

	glTexCoord2f(0.0f, texcoord);
	glVertex3fv(!(Cloud->Center + (Cloud->vy - Cloud->vx) * Cloud->Radius));

	glEnd();

	NumImpostors++;

	glDisable(GL_BLEND);
	glDisable(GL_ALPHA_TEST);

	glDisable(GL_TEXTURE_2D);
}

void VolumetricClouds::SortParticles(VolumetricCloud* Cloud, int mode)
{
	bool done = false, do_swap, useSTL = true;
	unsigned i, j;

	if (useSTL)
	{
		switch (mode)
		{
		case SORT_AWAY:
			sort(Cloud->Puffs.begin(), Cloud->Puffs.end(), SortAway);
			break;
		case SORT_TOWARD:
			sort(Cloud->Puffs.begin(), Cloud->Puffs.end(), SortToward);
			break;
		}
	}
	else
	{
		while (!done)
		{
			done = true;
			for (i = 0; i < Cloud->Puffs.size(); i++)
				for (j = i + 1; j < Cloud->Puffs.size(); j++)
				{
					do_swap = false;
					switch (mode)
					{
					case SORT_AWAY:
						if (Cloud->Puffs[i].DistanceToCam < Cloud->Puffs[j].DistanceToCam)
							do_swap = true;
						break;
					case SORT_TOWARD:
						if (Cloud->Puffs[i].DistanceToCam > Cloud->Puffs[j].DistanceToCam)
							do_swap = true;
						break;
					}
					if (do_swap)
					{
						swap(Cloud->Puffs[i], Cloud->Puffs[j]);
						done = false;
					}
				}
		}
	}
}

void VolumetricClouds::Destroy()
{
	unsigned i;
	for (i = 0; i < Clouds.size(); i++)
	{
		Clouds[i].Puffs.clear();
		delete[] Clouds[i].VertexBuffer;
		delete[] Clouds[i].ColorBuffer;
		delete[] Clouds[i].TexCoordBuffer;
	}

	Clouds.clear();

	glDeleteFramebuffersEXT(1, &g_framebuffer);
}

float VolumetricClouds::PhaseFunction(float omega)
{
	float g = 0.85;
	//return /*1.0/ (4.0 * M_PI) **/ 1.5*(1.0 - g*g)*(1+omega*omega) /(2+g*g)/powf(1.0 + g*g - 2.0 * g * omega, 1.5);

	return 0.75 * (1 + omega * omega)/*/(4*M_PI)*/;
	//return 1.0/(4*M_PI);
}

void VolumetricClouds::NormalizeExtinction()
{
	float scale_max = max_ext;
	for (int cloud_id = 0; cloud_id < Clouds.size(); cloud_id++)
	{
		int  Num = Clouds[cloud_id].Puffs.size();
		for (int i = 0; i < Num; i++)
		{
			min_ext = 0.0;
			float scale_ext = (min(Clouds[cloud_id].Puffs[i].Extinction, scale_max) - min_ext) / (scale_max - min_ext);
			scale_ext = 100 * (0.4 * scale_ext + 0.6);
			scale_ext = 65;
			Clouds[cloud_id].Puffs[i].Extinction = scale_ext;
		}
	}
}


