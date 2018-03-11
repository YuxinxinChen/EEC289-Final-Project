//each kernel process one node
__globle__ largeKernel(int *offset, int *col_id, int *large, int *set, int *color, int currentColor)
{
	//get the node from large array
	int node = large[blockIdx.x];
	int neighLen = offset[node+1]-offset[node];

	for(int i = threadIdx.x; i<neighLen; i=i+blockDim.x)
	{
		int item = col_id[offset[node]+i];
		if(item >= node && color[item]==0)
			set[node]=0;
	}
	__syncthreads();

	if(threadIdx.x == 0){
	if(set[node] == 1)
		color[node]=currentColor;
	}
}

	
