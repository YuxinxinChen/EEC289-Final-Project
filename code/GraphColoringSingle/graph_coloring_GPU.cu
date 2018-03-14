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
#include "cuda_query.cu"
#include "GraphColoringKernel.h"
#include "stopKernel.cu"

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

   //8)GraphColoring:let each thread compare one host vertex's value with one of its neighbor vertexes
   standard_context_t context;
   uint32_t sizeNode = NumRow;
   uint32_t sizeLbs = numNNZ;
   int blockSize = 512;
   int gridSize = (sizeLbs + blockSize -1) / blockSize;
   int* lbs;
   int* wir;
   HANDLE_ERROR(cudaMallocManaged(&lbs, numNNZ*sizeof(int)));
   HANDLE_ERROR(cudaMallocManaged(&wir, numNNZ*sizeof(int)));
   load_balance_search(sizeLbs, (int*)offset, sizeNode,lbs,context);
   cudaDeviceSynchronize();
   WorkItemRank<<<gridSize,blockSize>>>((int*)offset, lbs, wir, sizeLbs);
   cudaDeviceSynchronize();

   bool* setTrue;
   HANDLE_ERROR(cudaMallocManaged(&setTrue, NumRow*sizeof(bool)));
   memset(setTrue, 1, NumRow);

   int *shouldStop;
   HANDLE_ERROR(cudaMallocManaged(&shouldStop, 1*sizeof(int)));
   memset(shouldStop, 0, 1);
   
   int threadnum = 256;
   int blocknum = (NumRow + threadnum -1)/threadnum;

   int c = 1;
   while(*shouldStop == 0)
   {
	*shouldStop = 1;
	FindChangeColor<<<blocknum, threadnum>>>(wir, lbs, numNNZ, col_id, offset, c, color, setTrue);
//        GraphColoringKernel<<<blocknum,threadnum>>>(c, numNNZ, col_id, offset, lbs, wir, randoms, color, setTrue);
        cudaDeviceSynchronize();
	assignColor<<<blocknum, threadnum>>>(setTrue, NumRow, color, c);
//        ColorChanging<<<blocknum,threadnum>>>(c, NumRow, color, setTrue);
        cudaDeviceSynchronize();

	stopKernel<<<blocknum, threadnum>>>(color, shouldStop, NumRow);
	cudaDeviceSynchronize();
	c++;
    }

   printf("GraphColoringKernel found solution with %d colors\n", CountColors(V, color));
   printf("Valid coloring: %d\n", IsValidColoring(graph, V, color));
/***********************************************************************/
   //7) Color Vertices on CPU
//   GraphColoring(graph, V, &color);
//   printf("Brute-foce solution has %d colors\n", CountColors(V, color));   
//   printf("Valid coloring: %d\n", IsValidColoring(graph, V, color));
//
//   GreedyColoring(graph, V, &color);
//   printf("\n***************\n");
//   printf("Greedy solution has %d colors\n", CountColors(V, color));
//   printf("Valid coloring: %d\n\n", IsValidColoring(graph, V, color));
   //PrintSolution(color,V);
   /***********************************************************************/


   return 0;
}
