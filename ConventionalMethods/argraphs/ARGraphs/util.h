#ifndef __util_h
#define __util_h

//#define OUTPUT_MODE			1
//#define DEBUG_MODE		1

#define NO_MAX_THR			-1

#define INVALID				-100

#define	WINDOWS				1
#define LINUX				2

#define STOP				0
#define CONTINUE			1

#define OVERLAPPING			1
#define NONOVERLAPPING		0

#define EUCLIDEAN			1
#define CANBERRA			2

#define OBJECT_BASED		1
#define PIXEL_BASED			2

#define ZERO				0.00000000000000000000001
#define SMALL				0.00001
#define PI					3.14159265358979

#define BRIGHT				1
#define DARK_IN				2
#define DARK_OUT			3

#define THREE				3
#define FOUR				4
#define SIX					6
#define EIGHT				8
#define MORE				100
#define BOTH				2000

#define UNVISITED			0
#define VISITED				1
#define NOT_IN				2

#define TRUE				1
#define FALSE				0

#define BACKGROUND			0
#define UNSEGMENTED			0
#define MARKED_UNSEGMENTED	-1
#define TEMPORARY_MARKED	-2
#define TEMP_VALUE			-400
#define NO_OBJECT			-1

#define LUMEN				1
#define NUCLEI				2
#define IGNORE				3

#define LUMEN_LABEL			1
#define NONLUMEN_LABEL		2

#define NC_PIX				0
#define ST_PIX				1
#define LM_PIX				2
#define NC_VOID				3
#define ST_VOID				4
#define LM_VOID				5

#define NO_LABEL			-1
#define DONT_CARE			-2
#define NC_NC				1
#define ST_ST				2
#define LM_LM 				3
#define NC_ST 				4
#define NC_LM 				5
#define ST_LM 				6

#define MAX_DISTANCE 		0 
#define TOTAL_DISTANCE 		1

#define	S_SQUARE 			0
#define	S_OCTAGON 			1
#define S_DISK				2

#define CONSIDER_AREAS		0
#define DO_NOT_CONSIDER_AREAS 1

#define OBJPERC 			1
#define OBJAREA 			2
#define COLOR 				3
#define OBJPERC_OBJAREA 	4
#define OBJPERC_COLOR		5
#define OBJAREA_COLOR		6
#define OBJPERC_OBJAREA_COLOR 7

#define ISOLATED_NO 		1
#define ISOLATED_PERC		2

#define YES 				0
#define MAYBE 				1
#define NO 					2
#define TAKE_ALL 			3

#define RED 				0
#define BLUE 				1
#define GREEN 				2
#define YELLOW 				3

#define STRONG	 			2
#define WEAK	 			1
#define DUMMY				-999

#define AREA				1
#define CRIT				2

#define ALL_COMPONENTS		1
#define MEDIUM_COMPONENTS	2
#define AT_LEAST_1MEDIUM	3

#define GT3					1
#define SE3					2
#define ONLY3				3
#define ALL					4
#define GT3_NO_WEAK			5
#define SE3_NO_WEAK			6
#define ONLY2				7
#define REMAINING			8

#define SMALL_SIZE 			1
#define MEDIUM_SIZE 		2
#define LARGE_SIZE 			3

#define BR_QUAD				1
#define TR_QUAD				2
#define BL_QUAD				3
#define TL_QUAD				4

#define	HORIZONTAL			1
#define	VERTICAL			2

#define MISSING				1
#define CURRENT				2

#define MAX_SEQ				1000
#define MAX_LEN				200

#define RED_SOBEL 			1
#define GREEN_SOBEL 		1
#define BLUE_SOBEL 			2
#define YELLOW_SOBEL 		2

#define OFFSET				8

#define WHITE 				1
#define BLACK 				0

#define READ 				1
#define WRITE 				2

#define INNER_REG 			1
#define MID_REG 			2
#define OUTER_REG 			3

#define NONE 				-1
#define LAB 				1
#define RGB 				2

#define REPLICATE			1

#define UINT8				1
#define DOUBLE				2

#define DELETED -1
#define RIGHT 0
#define LEFT 1
#define BOTTOM 2
#define TOP 3

#define MINX 0
#define MAXX 1
#define MINY 2
#define MAXY 3

#define BOUNDARY_OFFSET		20

#define SQUARE(a) ( (a) * (a) )
#define SQRDIST(a,b) ( ((a)-(b)) * ((a)-(b)) )
#define SQRDISTP(x1,y1,x2,y2) (  ( ((x1)-(x2)) * ((x1)-(x2)) ) + ( ((y1)-(y2)) * ((y1)-(y2)) )  )
#define SWAP(a,b) {temp=(a); (a) = (b); (b) = temp;}

int indexOfMinArrayEntry(double A[], int size);
int maxArrayEntryIndex(int *A, int size);
int *readIntArrayFromFile(char *filename, int *no);
void writeDoubleArrayIntoFile(double *A, long size, char *filename);
void writeIntArrayIntoFile(int *A, long size, char *filename);
void sortDouble(double *A, int size, int *aOrder);
void sortLong(long *A, int size, int *aOrder);
double averageArray(double *A, int no);
double varianceArray(double *A, int no);

void terminateProgram(char *str);

int computeAngle(double x1, double y1, double x2, double y2, double *d);
int findMedian(int A[], int no);


struct THeapData {
	double key;
	int cx, cy;
	int label;
};
typedef struct THeapData HEAPDATA;

struct THeap{
	HEAPDATA *data;
	int maxSize;
	int size;
	int typ;
};
typedef struct THeap HEAP;

#define MAXHEAP 1
#define MINHEAP 2

HEAP initializeHeap(int maxSize, int typ);
void freeHeap(HEAP H);
void insertHeap(HEAP *H, double key, int cx, int cy, int label);
HEAPDATA deleteHeap(HEAP *H);


#endif


