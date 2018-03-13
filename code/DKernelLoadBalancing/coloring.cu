#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>

#include "getSortedDegree.cu"
#include "stopKernel.cu"
#include "largeKernel.cu"
#include "mediumKernel.cu"
#include "smallKernel.cu"

using namespace cub;

void coloring(int NumRow, int numNNZ, int *col_id, int *offset, int *color, bool *set)
{
        int currentColor = 1;

        int *shouldStop;
        HANDLE_ERROR(cudaMallocManaged(&shouldStop, 1*sizeof(int)));
        memset(shouldStop, 0, 1);

       	// numThreads=512 might works well TODO
        int numThreads(512), numBlocks((NumRow+numThreads-1)/numThreads);
	//get an array of degree of every node, then sort the array, group them into CTA, warp, thread
	int *workspace;
	HANDLE_ERROR(cudaMallocManaged(&workspace, 4*NumRow*sizeof(int)));
	getSortedDegree<<<numBlocks, numThreads>>>(NumRow, offset, workspace, workspace+NumRow, workspace+2*NumRow);
        cudaDeviceSynchronize();

	void  *d_temp_storage = NULL;
        size_t    temp_storage_bytes = 0;
        DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, workspace, workspace+3*NumRow, NumRow);
        HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, workspace, workspace+3*NumRow, NumRow);
        cudaDeviceSynchronize();

	int sizeLarge = workspace[3*NumRow+NumRow-1];
	int *Large;
	HANDLE_ERROR(cudaMallocManaged(&Large, sizeLarge*sizeof(int)));
	filter<<<numBlocks, numThreads>>>(workspace, workspace+3*NumRow, Large, NumRow);
	cudaDeviceSynchronize();
	std::cout<<"sizeLarge="<<sizeLarge<<std::endl;
		
	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, workspace+NumRow, workspace+3*NumRow, NumRow);
	cudaDeviceSynchronize();

	int sizeMedium = workspace[3*NumRow+NumRow-1];
	int *Medium;
	HANDLE_ERROR(cudaMallocManaged(&Medium, sizeMedium*sizeof(int)));
	filter<<<numBlocks, numThreads>>>(workspace+NumRow, workspace+3*NumRow, Medium, NumRow);
	cudaDeviceSynchronize();
	std::cout<<"sizeMedium = "<<sizeMedium<<std::endl;

	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, workspace+2*NumRow, workspace+3*NumRow, NumRow);
	cudaDeviceSynchronize();

	int sizeSmall = workspace[3*NumRow+NumRow-1];
	int *Small;
	HANDLE_ERROR(cudaMallocManaged(&Small, sizeSmall*sizeof(int)));
	filter<<<numBlocks, numThreads>>>(workspace+2*NumRow, workspace+3*NumRow, Small, NumRow);
	cudaDeviceSynchronize();	
	std::cout<<"sizeSmall="<<sizeSmall<<std::endl;

	while(*shouldStop == 0)
	{
	   *shouldStop = 1;
	   // process nodes with large neighbor list with CTA
	   largeKernel<<<sizeLarge, numThreads>>>(offset, col_id, Large, sizeLarge, color, currentColor);
	   cudaDeviceSynchronize();
	   // process nodes with median neighbor list with warp
	   mediumKernel<<<(sizeMedium + numThreads/32 -1 )/(numThreads/32), numThreads, numThreads/32*sizeof(bool)>>>(offset, col_id, Medium, sizeMedium, color, currentColor);
	   cudaDeviceSynchronize();
	   // process nodes with small neighbor list with thread
	   smallKernel<<<(sizeSmall+numThreads-1)/numThreads, numThreads>>>(offset, col_id, Small, sizeSmall, color, currentColor);
	   cudaDeviceSynchronize();

	   // check if all the nodes are colored
	   stopKernel<<<numBlocks, numThreads>>>(color, shouldStop, NumRow);
	   cudaDeviceSynchronize();
	
	   currentColor++;
	}	
}	
