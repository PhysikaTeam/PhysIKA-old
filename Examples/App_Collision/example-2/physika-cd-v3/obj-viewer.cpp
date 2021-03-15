#include <stdio.h>

//rky
#include "CollisionAPI.h"
//rky

float DISP_SCALE = 0.001f;
char *dataPath;
int stFrame = 0;
float p[3], q[3];

static int level = 1;

float lightpos[4] = {13, 10.2, 3.2, 0};

// check for OpenGL errors

extern void drawEdges(bool, bool);
extern void drawVFs(int);
extern void drawDebugVF(int);
extern void initModel(int, char **, char *, int);

static bool ret = false;


extern mesh *cloths[16];
extern mesh *lions[16];

int main(int argc, char **argv)
{
	if (argc < 2)
		printf("usage: %s data_path [start_frame] [device=N]\n", argv[0]);

	dataPath = argv[1];

	if (argc == 3) {
		sscanf(argv[2], "%d", &stFrame);
	}

	initModel(argc, argv, dataPath, stFrame);


	Collision sc;
	for (int i = 0; i < 16; i++){
		if (cloths[i] != NULL){
			sc.Prepare(cloths[i],true);
			//sc.Prepare(cloths[i], true, lions[0], false);
		}
			
	}
	//sc.setflag(CollisionFlag::USE_CPU_REFIT);
	sc();

	

	vector<vector<tri_pair>> tem = sc.getContactPairs();
	unsigned int id[4];
	for (int i = 0; i < sc.getNumContacts(); i++){
		tem[i][0].get(id[0], id[1]);
		tem[i][1].get(id[2], id[3]);
		//printf("%d %d %d %d\n", id[0], id[1], id[2], id[3]);
	}
	
	printf("%d\n", sc.getNumContacts());
	return 0;
}


