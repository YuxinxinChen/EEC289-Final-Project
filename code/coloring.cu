#include "assign_color_gl.cu"
#include "indept_set_gl.cu"

#define NUM_COLOR_PER_THREAD 1 //this is changed to be more than one, then we need to move my_offset_start and my_offset_end to be in shared memory instead 

void coloring(uint32_t NumRow, uint32_t numNNZ, uint32_t *col_id, uint32_t *offset, int *color, bool *set)
{
	int currentColor = 1;
	
	int *numColored;
   	HANDLE_ERROR(cudaMallocManaged(&numColored, 1*sizeof(int)));
   	memset(set, 0, 1);
	
       int numBlocks(1), numThreads(1);
       if(NumRow < 1024){//if it is less than 1024 vertex, then launch one block 
         numBlocks = 1;
         numThreads = 1024;
       }else{//otherwise, launch as many as 1024-blocks as you need         
         numBlocks = (NumRow+1023)/1024;
         numThreads = 1024;
       }


	bool useMax = 1;
	while(*numColored < NumRow){
		indept_set_gl<<<numBlocks, numThreads>>>(NumRow, col_id, offset, set, useMax, numColored);
		cudaDeviceSynchronize();
		assign_color_gl<<<numBlocks, numThreads>>>(currentColor,NumRow, set, color);
		cudaDeviceSynchronize();
		currentColor++;
	}
}
	
