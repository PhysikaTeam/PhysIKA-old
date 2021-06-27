#pragma once

#include "elementload.h" 

int readKFile(FEMDynamic *femAnalysis, string inFile);
int readINPFile(FEMDynamic *femAnalysis, string inFile);
void readCommand(int argc, char* argv[], string &inFile, string &outFile, FEMDynamic *dy);
void history_record_start(string fin, string fout1, char exeFile[]);
void history_record_end(double calculae_time, double contact_time, double fileTime, int calFlag, int iterNum, int gpuNum);
void copyrightStatement();
void AISIMExplicitCopyrightStatement();
int connection_for_fem(FEMDynamic *femAnalysis);