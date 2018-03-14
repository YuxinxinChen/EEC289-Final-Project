__global__ void groupDegree(int *offset, int NumNodes, int *workspace1, int *workspace2, int *neighLen, int threshold)
{
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<NumNodes; i+=gridDim.x*blockDim.x)
	{
		// init workspace to 0
		workspace1[i]=0;
		workspace2[i]=0;

		// get neighbor list lenght for each node and put it into workspace1 if it is small, workspace2 otherwise
		int len = offset[i+1]-offset[i];
		if(len <= threshold)
		workspace1[i]= 1;
		else workspace2[i]=1;

		neighLen[i] = len;
	}
}

__global__ void filter(int *predicateArray, int* scanArray, int *newPlace, int sizeScan)
{
        for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<sizeScan; i=i+gridDim.x*blockDim.x)
        {
                if(predicateArray[i] ==1)
                        newPlace[scanArray[i]-1] = i;
        }
}

__global__ void filter2(int *predicateArray, int* scanArray, int *neighLen, int *newPlace, int*newNeighLen, int sizeScan)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<sizeScan; i+=gridDim.x*blockDim.x)
	{
		if(predicateArray[i] == 1)
		{
			newPlace[scanArray[i]-1] = i;
			newNeighLen[scanArray[i]-1] = neighLen[i];
		}
	}
}

		
