#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"
#include "../util.h"
#include "markerDefinition.h"
#include "watershed.h"
//#include "vld.h"

int parseMainArguments(int argc, char *argv[]){
    //printf("Anzahl der Argumente: %d",argc);
	if (argc < 2)
		terminateProgram("Enter\n\t1: Marker definition\n\t2: Flooding");

/*	if (atoi(argv[1]) != 1 && atoi(argv[1]) != 2)
		terminateProgram("Enter\n\t1: Marker definition\n\t2: Flooding");
*/	
	return atoi(argv[1]);
    return 0;
}
int ARGraphMex(int argc, char *argv[]){
	int choice = parseMainArguments(argc,argv);
    int j=0;
   // printf("Choice: %d",choice);
	//time_t start, finish;
	//time( &start );
/* for( j=0; j<argc; j++ )
 {
        mexPrintf("Input %d = <%s>\n",j,argv[j]);
 }
  */  
	if (choice == 1)
		MARKER_DEFINITION(argc,argv);
	else
		REGION_GROWING(argc,argv);
   
	//time( &finish );
	//printf( "\nProgram takes %6.0f seconds.\n", difftime( finish, start ) );

	return 0;
}
void mexFunction(
    int		  nlhs, 	/* number of expected outputs */
    mxArray	  *plhs[],	/* mxArray output pointer array */
    int		  nrhs, 	/* number of inputs */
    const mxArray	  *prhs[]	/* mxArray input pointer array */
    )
    {
        int argc = 0;
        char **argv;
        mwIndex i;
        int k, ncell;
        int j = 0;

        for( k=0; k<nrhs; k++ )
        {
            if( mxIsCell( prhs[k] ) )
            {
                argc += ncell = mxGetNumberOfElements( prhs[k] );
                for( i=0; i<ncell; i++ )
                    if( !mxIsChar( mxGetCell( prhs[k], i ) ) )
                        mexErrMsgTxt("Input cell element is not char");
            }
            else
            {
                argc++;
                if( !mxIsChar( prhs[k] ) )
                    mexErrMsgTxt("Input argument is not char");
            }
        }

        argv = (char **) mxCalloc( argc+1, sizeof(char *) );
        argv[0] = "";
        for( k=0; k<nrhs; k++ )
        {
            if( mxIsCell( prhs[k] ) )
            {
                ncell = mxGetNumberOfElements( prhs[k] );
                for( i=0; i<ncell; i++ )
                    argv[1+j++] = mxArrayToString( mxGetCell( prhs[k], i )
    );
            }
            else
            {
                argv[1+j++] = mxArrayToString( prhs[k] );
            }
        }
        argc++;
    ARGraphMex(argc,argv);
}