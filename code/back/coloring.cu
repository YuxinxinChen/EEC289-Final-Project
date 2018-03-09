#include "assign_color.cu"
#include "indept_set.cu"

#define NUM_COLOR_PER_THREAD 1 //this is changed to be more than one, then we need to move my_offset_start and my_offset_end to be in shared memory instead 

__device__ int numColored = 0;

__global__ void graphColoring(uint32_t NumRow, //number of vertices (= number of rows in adjacency matrix)
	                     uint32_t numNNZ, //number of non zero entry of the adjacency matrix
	                     uint32_t *col_id, //the column id in the CSR format 
	                     uint32_t *offset, //the row offset in the CSR
	                     int* color, //the color of the vertices (output)
	                     bool*set //the indepent set (global memory)
	                     ){

	int currentColor = 1;
	

	while(numColored < NumRow){//loop untill all vertices are colored 

		indept_set(NumRow, numNNZ, col_id, offset, set, currentColor%2 == 1, color, numColored);
		__syncthreads();		
		assign_color(currentColor, NumRow, set,color);
		__syncthreads();
		currentColor++;	
	}		
}
