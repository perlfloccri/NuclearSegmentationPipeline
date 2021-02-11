#ifndef __markerDefinition_h
#define __markerDefinition_h


#include "../matrix.h"
#include "sobelPrimitives.h"
#include "sobelTasks.h"
#include "cellCandidates.h"

MATRIX topSobels, bottomSobels, rightSobels, leftSobels;
int writeSobels;

void allocateGlobalSobels(int dx, int dy);
void freeGlobalSobels();
void writeGlobalSobels(char *fname);
void parseArguments(int argc, char *argv[], int *smallThr, int *critType, double *otsuPercMin, double *critInitThr, int *cid);
void writeBoundariesIntoFile(MATRIX res, char *fname);
void MARKER_DEFINITION(int argc, char *argv[]);

#endif



