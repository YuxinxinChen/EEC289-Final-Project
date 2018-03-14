#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>

#include "groupDegree.cu"
#include "stopKernel.cu"
#include "smallKernel.cu"
#include "lbs.cu"

using namespace cub;
using namespace mgpu;

void coloring(int NumRow, int numNNZ, int *col_id, int *offset, int *color)
{
	standard_context_t context;
        int currentColor = 1;

        int numThreads(512), numBlocks((NumRow+numThreads-1)/numThreads);
	// get neighlist len array and produce predicates for small group and large group
	int *workspace;
	HANDLE_ERROR(cudaMallocManaged(&workspace, 4*NumRow*sizeof(int)));
	int *neighLen;
	HANDLE_ERROR(cudaMallocManaged(&neighLen, NumRow*sizeof(int)));

	groupDegree<<<numBlocks, numThreads>>>(offset, NumRow, workspace, workspace+NumRow, neighLen, 64);
        cudaDeviceSynchronize();

		
	// filter out the nodes with small degree 
	void  *d_temp_storage = NULL;
        size_t    temp_storage_bytes = 0;
        DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, workspace, workspace+2*NumRow, NumRow);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, workspace, workspace+2*NumRow, NumRow);
        cudaDeviceSynchronize();

	int sizeSmall = workspace[2*NumRow+NumRow-1];
	int *Small;
	if(sizeSmall > 0) {
	   HANDLE_ERROR(cudaMallocManaged(&Small, sizeSmall*sizeof(int)));
	   filter<<<numBlocks, numThreads>>>(workspace, workspace+2*NumRow, Small, NumRow);
	   cudaDeviceSynchronize();	
	   std::cout<<"sizeSmall="<<sizeSmall<<std::endl;
	}

	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, workspace+NumRow, workspace+2*NumRow, NumRow);
	cudaDeviceSynchronize();
	// filter out the ndoes with large degree
	int sizeLarge = workspace[NumRow+NumRow-1];
	int *Large(NULL), *newNeighLen(NULL), *scanNeighLen(NULL);
	int *lbs(NULL), *wir(NULL);
	int *set(NULL);
	int sizeLbs = 0;
	if(sizeLarge > 0){
	   HANDLE_ERROR(cudaMallocManaged(&Large, sizeLarge*sizeof(int)));
	   HANDLE_ERROR(cudaMallocManaged(&newNeighLen, sizeLarge*sizeof(int)));
	   filter2<<<numBlocks, numThreads>>>(workspace+NumRow, workspace+2*NumRow, neighLen, Large, newNeighLen, NumRow);
	   cudaDeviceSynchronize();
	   std::cout<<"sizeLarge="<<sizeLarge<<std::endl;
	   HANDLE_ERROR(cudaMallocManaged(&scanNeighLen, (sizeLarge+1)*sizeof(int)));
	   DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, newNeighLen, scanNeighLen, sizeLarge+1);
	   HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	   
	   DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, newNeighLen, scanNeighLen, sizeLarge+1);
	   cudaDeviceSynchronize();

	   sizeLbs = scanNeighLen[sizeLarge];

	   HANDLE_ERROR(cudaMallocManaged(&lbs, sizeLbs*sizeof(int)));
	   HANDLE_ERROR(cudaMallocManaged(&wir, sizeLbs*sizeof(int)));

	   load_balance_search(sizeLbs, scanNeighLen, sizeLarge, lbs, context);
	   cudaDeviceSynchronize();

	   WorkItemRank<<<(sizeLarge+511)/512, 512>>>(scanNeighLen, lbs, wir, sizeLbs);
	   cudaDeviceSynchronize();

           HANDLE_ERROR(cudaMallocManaged(&set, sizeLarge*sizeof(int)));
	   memset(set, 1, sizeLarge);
	   
	}	
	// run large degree nodes array with lbs
	// run small degree nodes with smallKernel

        int *shouldStop;
        HANDLE_ERROR(cudaMallocManaged(&shouldStop, 1*sizeof(int)));
        memset(shouldStop, 0, 1);


	while(*shouldStop == 0)
	{
	   *shouldStop = 1;
	   // process nodes with large neighbor list with CTA
	   if(sizeLarge > 0) {
		FindChangeColor<<<(sizeLarge+511)/512, 512>>>(sizeLarge, Large, wir, lbs, sizeLbs, col_id, offset, currentColor, color, set);
		cudaDeviceSynchronize();
		assignColor<<<(sizeLarge+511)/512, 512>>>(set, sizeLarge, Large, color, currentColor);
		cudaDeviceSynchronize();
	   }
	   // process nodes with small neighbor list with thread
	   if(sizeSmall > 0 ) {
	   	smallKernel<<<(sizeSmall+numThreads-1)/numThreads, numThreads>>>(offset, col_id, Small, sizeSmall, color, currentColor);
	   	cudaDeviceSynchronize();
	   }

	   // check if all the nodes are colored
	   stopKernel<<<numBlocks, numThreads>>>(color, shouldStop, NumRow);
	   cudaDeviceSynchronize();
	
	   currentColor++;
	}
}	
