#include <fstream>
#include "windows.h"
using namespace std;

class timer
{
public:
	timer(void);
    float  start;  //ms
	float  end;    //ms
	DWORD  elapse; //ms
	char eventName[100];

    void Begin(char* eventname)
	{
		LARGE_INTEGER nFreq, nBefore;
		memset(&nFreq,   0x00, sizeof nFreq);
		memset(&nBefore, 0x00, sizeof nBefore);
		QueryPerformanceFrequency(&nFreq);
		QueryPerformanceCounter(&nBefore); 
	    start =nBefore.QuadPart *1000.0/ nFreq.QuadPart;
		strcpy(eventName,eventname);

	}
	void End()
	{
		LARGE_INTEGER nFreq,  nAfter;
		memset(&nFreq,   0x00, sizeof nFreq);
		memset(&nAfter, 0x00, sizeof nAfter);
		QueryPerformanceFrequency(&nFreq);
		QueryPerformanceCounter(&nAfter); 
		end =nAfter.QuadPart *1000.0/ nFreq.QuadPart;	
		elapse=DWORD(end-start);

	}
	DWORD GetElapse()
	{
		return elapse;
	}
	void OutElapse2File(ofstream& out)
	{
		out<<eventName<<": " <<elapse<<"ms"<<endl;
		out.flush();
	}


public:
	~timer(void);
};
