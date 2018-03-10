#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>

#include "getSortedDegree.cu"

using namespace cub;

void coloring(int NumRow, int numNNZ, int *col_id, int *offset, int *color, bool *set)
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

	int *test_offset;
	int test_NumRow = 10;
	HANDLE_ERROR(cudaMalloc(&test_offset, test_NumRow*sizeof(int)));
	test_offset[0]=0;
	test_offset[1]=513;
	test_offset[2]=513+3;
	test_offset[3]=513+3+31;
	test_offset[4]=513+3+31+35;
	test_offset[5]=513+3+31+35+10;
	test_offset[6]=513+3+31+35+10+514;
	test_offset[7]=513+3+31+35+10+514+40;
	test_offset[8]=513+3+31+35+10+514+40+128;
	test_offset[9]=513+3+31+35+10+514+40+129+28;


	//get an array of degree of every node, then sort the array, group them into CTA, warp, thread
	int *workspace;
	HANDLE_ERROR(cudaMallocManaged(&workspace, 4*test_NumRow*sizeof(int)));
	getSortedDegree<<<numBlocks, numThreads>>>(test_NumRow, test_offset, workspace, workspace+test_NumRow, workspace+2*test_NumRow);
        cudaDeviceSynchronize();
	
	void  *d_temp_storage = NULL;
        size_t    temp_storage_bytes = 0;
        DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, workspace, workspace+3*test_NumRow, test_NumRow);
        HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, workspace, workspace+3*test_NumRow, test_NumRow);
        cudaDeviceSynchronize();
		
	int sizeLarge = workspace[test_NumRow-1];
	int *Large;
	HANDLE_ERROR(cudaMalloc(&Large, sizeLarge*sizeof(int)));
	
	filter<<<numBlocks, numThreads>>>(workspace, workspace+3*test_NumRow, Large, test_NumRow);
	
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, workspace+test_NumRow, workspace+3*test_NumRow, test_NumRow);
	cudaDeviceSynchronize();

	int sizeMediun = workspace[test_NumRow+test_NumRow-1];
	int *Medium;
	HANDLE_ERROR(cudaMalloc(&Medium, sizeMedium*sizeof(int)));
	
	filter<<<numBlocks, numThreads>>>(workspace+test_NumRow, workspace+3*test_NumRow, Medium, test_NumRow);
	cudaDeviceSynchronize();

	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, workspace+2*test_NumRow, workspace+3*test_NumRow, test_NumRow);
	
	int sizeSmall = workspace[2*test_NumRow+test_NumRow-1];
	int *Small;
	HANDLE_ERROR(cudaMalloc(&Small, sizeSmall*sizeof(int)));
	
	filter<<<numBlocks, numThreads>>>(workspace+2*test_NumRow, workspace+3*test_NumRow, Small, test_NumRow);
	cudaDeviceSynchronize();	

	for(int i=0; i<sizeLarge; i++)
		std::cout<< Large[i]<<" ";
	std::cout<<std::endl;

	for(int i=0; i<sizeMedium; i++)
		std::cout<< Medium[i]<<" ";
	std::cout<<std::endl;

	for(int i=0; i<sizeSmall; i++)
		std::cout<<Small[i]<<" ";
	std::cout<<std::endl;

	// process nodes with large neighbor list with CTA
	// process nodes with median neighbor list with warp
	// process nodes with small neighbor list with thread
}
