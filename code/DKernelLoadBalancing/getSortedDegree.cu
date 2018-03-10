__global__ void getSortedDegree(int numNodes,  int *offset, int *workspace1, int *workspace2, int *workspace3)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<numNodes; i++)
	{
		// initiate all workspace to 0
		workspace1[i] = 0;
		workspace2[i] = 0;
		workspace3[i] = 0;

		// compute each neighlist's length
		int neighlistLen=offset[i+1]-offset[i];

		// group the nodes by their degree
		if(neighlistLen >= 512) workspace1[i] = 1;
		else if(neighlistLen > 32) workspace2[i] = 1;
		else if(neighlistLen >0 && neighlistLen <= 32) workspace3[i] = 1;
	}
}

__global__ void filter(int *predicateArray, int* scanArray, int *newPlace, int sizeScan)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<sizeScan; i++)
	{
		if(predicateArray[i] ==1)
			newPlace[scanArray[i]-1] = i;
	}
}
	
