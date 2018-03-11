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

       	// numThreads=512 might works well TODO
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
	HANDLE_ERROR(cudaMallocManaged(&test_offset, test_NumRow*sizeof(int)));
	test_offset[0]=0;
	test_offset[1]=513;
	test_offset[2]=513+3;
	test_offset[3]=513+3+31;
	test_offset[4]=513+3+31+35;
	test_offset[5]=513+3+31+35+10;
	test_offset[6]=513+3+31+35+10+514;
	test_offset[7]=513+3+31+35+10+514+40;
	test_offset[8]=513+3+31+35+10+514+40+128;
	test_offset[9]=513+3+31+35+10+514+40+128+28;
	test_offset[10]=513+3+31+35+10+514+40+128+28+25;


	//get an array of degree of every node, then sort the array, group them into CTA, warp, thread
	int *workspace;
	HANDLE_ERROR(cudaMallocManaged(&workspace, 4*test_NumRow*sizeof(int)));
	getSortedDegree<<<numBlocks, numThreads>>>(test_NumRow, test_offset, workspace, workspace+test_NumRow, workspace+2*test_NumRow);
        cudaDeviceSynchronize();

	for(int i=0; i<test_NumRow; i++)
		std::cout<<workspace[i]<<" ";
	std::cout<<std::endl;
	
	for(int i=0; i<test_NumRow; i++)
		std::cout<<workspace[test_NumRow+i]<<" ";
	std::cout<<std::endl;

	for(int i=0; i<test_NumRow; i++)
		std::cout<<workspace[2*test_NumRow+i]<<" ";
	std::cout<<std::endl;

	void  *d_temp_storage = NULL;
        size_t    temp_storage_bytes = 0;
        DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, workspace, workspace+3*test_NumRow, test_NumRow);
        HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, workspace, workspace+3*test_NumRow, test_NumRow);
        cudaDeviceSynchronize();

	for(int i=0; i<test_NumRow; i++)
		std::cout<<workspace[3*test_NumRow+i]<<" ";
	std::cout<<std::endl;
		
	int sizeLarge = workspace[3*test_NumRow+test_NumRow-1];
	int *Large;
	HANDLE_ERROR(cudaMallocManaged(&Large, sizeLarge*sizeof(int)));
	
	filter<<<numBlocks, numThreads>>>(workspace, workspace+3*test_NumRow, Large, test_NumRow);
	cudaDeviceSynchronize();

	std::cout<<"sizeLarge="<<sizeLarge<<std::endl;
	for(int i=0; i<sizeLarge; i++)
		std::cout<<Large[i] << " ";
	std::cout<<std::endl;
	
	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, workspace+test_NumRow, workspace+3*test_NumRow, test_NumRow);
	cudaDeviceSynchronize();

	for(int i=0; i<test_NumRow; i++)
		std::cout<<workspace[3*test_NumRow+i]<<" ";
	std::cout<<std::endl;

	int sizeMedium = workspace[3*test_NumRow+test_NumRow-1];
	int *Medium;
	HANDLE_ERROR(cudaMallocManaged(&Medium, sizeMedium*sizeof(int)));
	
	filter<<<numBlocks, numThreads>>>(workspace+test_NumRow, workspace+3*test_NumRow, Medium, test_NumRow);
	cudaDeviceSynchronize();

	std::cout<<"sizeMedium = "<<sizeMedium<<std::endl;
	for(int i=0; i<sizeMedium; i++)
		std::cout<<Medium[i] << " ";
	std::cout<<std::endl;

	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, workspace+2*test_NumRow, workspace+3*test_NumRow, test_NumRow);
	cudaDeviceSynchronize();

	for(int i=0; i<test_NumRow;i++)
		std::cout<<workspace[3*test_NumRow+i]<<" ";
	std::cout<<std::endl;
	
	int sizeSmall = workspace[3*test_NumRow+test_NumRow-1];
	int *Small;
	HANDLE_ERROR(cudaMallocManaged(&Small, sizeSmall*sizeof(int)));
	
	filter<<<numBlocks, numThreads>>>(workspace+2*test_NumRow, workspace+3*test_NumRow, Small, test_NumRow);
	cudaDeviceSynchronize();	

	std::cout<<"sizeSmall="<<sizeSmall<<std::endl;
	for(int i=0; i<sizeSmall; i++)
		std::cout<<Small[i]<<" ";
	std::cout<<std::endl;

	// process nodes with large neighbor list with CTA
	// process nodes with median neighbor list with warp
	// process nodes with small neighbor list with thread
}
