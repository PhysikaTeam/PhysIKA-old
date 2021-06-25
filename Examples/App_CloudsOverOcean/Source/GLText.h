
#ifndef GL_TEXT_H
#define GL_TEXT_H

class GLText
{
public:

	GLText();

	int		LoadFont(char *fname, int ScrWidth, int ScrHeight);
	void	Destroy();

	void TextOut(int x, int y, char* format, ...);
	void SetColor(Color4 color)
	{
		Color = color;
	}

//	int  GetPixelLength(string*);
//	int  GetHeight() { return fontHeight[curFont]; }

private:

	void DrawChar(unsigned char c, float x, float y);

	unsigned fontTexture;
	unsigned char charSpace[256];
	unsigned char fontHeight;

	Color4	Color;
	
	int ScreenWidth, ScreenHeight;
	
	int cursorPos;
};

#endif
