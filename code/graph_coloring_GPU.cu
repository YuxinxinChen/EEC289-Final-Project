#include <sstream>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//Error handling micro, wrap it around function whenever possible
static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		//system("pause");
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#include "validate.h"
#include "serial.h"
#include "utility.h"
#include "coloring.cu"
#include "cuda_query.cu"


int main(int argc, char* argv[])
{
	cuda_query(0); //Set the deivde number here 

	if(argc != 2){
		std::cout<<"  Usage ./graphGPU INPUTFILE"<<std::endl;
		std::cout<<"input files can be found under input/ "<<std::endl;
		exit(EXIT_FAILURE);
	}

   bool* graph;
   int V;  
   uint32_t numNNZ=0;
   uint32_t NumRow=0; //same as V


   //1) Read graph
   if (std::string(argv[1]).find(".col") != std::string::npos){
     ReadColFile(argv[1], &graph, &V, &numNNZ,&NumRow);
   } else if (std::string(argv[1]).find(".mm") != std::string::npos){
     ReadMMFile(argv[1], &graph, &V, &numNNZ,&NumRow);
   } else{
   	std::cout<<" Invalid file formate!!"<<std::endl;
   	exit(EXIT_FAILURE);
   }
   /***********************************************************************/

   //2) Allocate memory (on both sides)
   uint32_t *col_id(NULL),*offset(NULL);   
   HANDLE_ERROR(cudaMallocManaged(&col_id, numNNZ*sizeof(uint32_t)));
   
   //last entry will be = numNonZero (so that we have always a pointer
   //to the first and last id for each row with no need for if statments)   
   HANDLE_ERROR(cudaMallocManaged(&offset, (NumRow +1)*sizeof(uint32_t)));
   /***********************************************************************/

   //3) Get graph in CSR format 
   getCSR(numNNZ, NumRow, graph, col_id, offset);
   //printCSR(numNNZ,NumRow,col_id, offset);
   /***********************************************************************/

   //5) Color Vertices in paralllel
   int* color;
   HANDLE_ERROR(cudaMallocManaged(&color, NumRow*sizeof(int)));
   memset(color, 0, NumRow );   

   bool*set;
   HANDLE_ERROR(cudaMallocManaged(&set, NumRow*sizeof(bool)));
   memset(set, 0, NumRow); 

   coloring(NumRow, numNNZ, col_id, offset, color, set);
   
   //6) Validate parallel solution 
   printf("Parallel solution has %d colors\n", CountColors(V, color));
   printf("Valid coloring: %d\n\n", IsValidColoring(graph, V, color));
   //PrintSolution(color,V);
   /***********************************************************************/


   //7) Color Vertices on CPU
   GraphColoring(graph, V, &color);
   printf("Brute-foce solution has %d colors\n", CountColors(V, color));   
   printf("Valid coloring: %d\n", IsValidColoring(graph, V, color));

   GreedyColoring(graph, V, &color);
   printf("\n***************\n");
   printf("Greedy solution has %d colors\n", CountColors(V, color));
   printf("Valid coloring: %d\n\n", IsValidColoring(graph, V, color));
   //PrintSolution(color,V);
   /***********************************************************************/


   //8)GraphColoring:let each thread compare one host vertex's value with one of its neighbor vertexes
   standard_context_t context;
   int sizeNode = NumRow +1;
   int sizeLbs = offset[sizeNode];
   int *lbs(NULL), *wir(NULL);
   HANDLE_ERROR(cudaMallocManaged(&lbs, sizeLbs*sizeof(int)));
   HANDLE_ERROR(cudaMallocManaged(&wir, sizeLbs*sizeof(int)));
   load_balance_search(sizeLbs, offset, sizeNode, lbs, context);
   cudaDeviceSynchronize();
   WorkItemRank<<<gridSize,blockSize>>>(offset, lbs, wir, sizeLbs);
   cudaDeviceSynchronize();

   for(int c = 1; c < 254; c++) 
   {
        int threadnum = 256;
        int blocknum = V / threadnum + 1;
        GraphColoringKernel<<<blocknum,threadnum>>>(c, col_id, offset, bls, wir, randoms, color);
        cudaDeviceSynchronize();
    }

   printf("GraphColoringKernel found solution with %d colors\n", CountColors(V, color));
   printf("Valid coloring: %d\n", IsValidColoring(graph, V, color));
   /***********************************************************************/

   return 0;
}
