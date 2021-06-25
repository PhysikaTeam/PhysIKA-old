#include "Common.h"
GLText::GLText()
{
	Color.R = Color.G = Color.B = 0.0f; Color.A = 1.0f;
}

int GLText::LoadFont(char* fname, int Width, int Height)
{
	ScreenWidth = Width;
	ScreenHeight = Height;
	
	unsigned img;
	ilGenImages(1, &img);
	ilBindImage(img);
	if (ilLoadImage(ILstring(fname)) == false)
	{
		printf("Cannot load image file %s - %d\n", fname, ilGetError());
		return 1;
	}
	
	glGenTextures(1, &fontTexture);
	glBindTexture(GL_TEXTURE_2D, fontTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ilGetInteger(IL_IMAGE_WIDTH), ilGetInteger(IL_IMAGE_HEIGHT), 0, ilGetInteger(IL_IMAGE_FORMAT), GL_UNSIGNED_BYTE, ilGetData());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	
	ilDeleteImages(1, &img);
	
	char data[256];
	memset(data, 0, 256);
	strcpy(data, fname);
	int len = strlen(data);
	
	data[len] = 0;
	data[len - 1] = 't';
	data[len - 2] = 'a';
	data[len - 3] = 'd';
	
	FILE* f = 0;
	f = fopen(data, "rb");
	if (!f)
	{
		printf("Cannot load character spacing file %s", data);
		return 1;
	}

	fread(charSpace, 1, 256, f);
	fclose(f);
	
	return 0;
}

void GLText::DrawChar(unsigned char c, float x, float y)
{	
	float cPosX, cPosY;

	float texSz = 256.0f;

	float origX = 0; 
	float origY = 0; 
	float sizeX = texSz / 16;
	float sizeY = texSz / 16;

	unsigned char charSize = charSpace[c];

	y = y + (texSz / 16);

	cPosX = (c % 16) * (texSz / 16);
	cPosY = (c / 16) * (texSz / 16);
	
	glBegin(GL_QUADS);
		glTexCoord2f((cPosX + origX + 8 - charSize / 2 - 1) / texSz, (texSz - cPosY - origY) / texSz);
		glVertex2f(cursorPos, y - 16);
		
		glTexCoord2f((cPosX + origX + 8 + charSize / 2) / texSz, (texSz - cPosY - origY) / texSz);
		glVertex2f(cursorPos + charSize, y - 16);
		
		glTexCoord2f((cPosX + origX + 8 + charSize / 2) / texSz, (texSz - cPosY - origY - sizeY) / texSz);
		glVertex2f(cursorPos + charSize, y);
		
		glTexCoord2f((cPosX + origX + 8 - charSize / 2 - 1) / texSz, (texSz - cPosY - origY - sizeY) / texSz);
		glVertex2f(cursorPos, y);
	glEnd();

	cursorPos += charSize;
}

/*int  GLText::GetPixelLength(string *str)
{
	int res = 0;
	int i = 0;
	while (i < str->length())
	{
		res += charSpace[curFont][(*str)[i]];
		i++;
	}
	return res;
}*/

void GLText::TextOut(int x, int y, char* format, ...)
{	
	if (format == NULL)				
		return;

	char strText[1024];
	va_list args;

	va_start(args, format);
	vsprintf(strText, format, args);
	va_end(args);
		
	cursorPos = x;

	glMatrixMode(GL_MODELVIEW);	
	glPushMatrix(); 
	glLoadIdentity();		

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, ScreenWidth, ScreenHeight, 0);
	
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, fontTexture);
	
	glColor4fv(!Color);

	int i = 0;
	while (strText[i++] != 0)
	{
		DrawChar(strText[i - 1], x, y + 12);
	}

	glDisable(GL_BLEND);

	glDisable(GL_TEXTURE_2D);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix(); 
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix(); 
}


void GLText::Destroy()
{
	glDeleteTextures(1, &fontTexture);
}
