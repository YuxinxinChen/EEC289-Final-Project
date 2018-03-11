//numThreads should be multiple of 32
__global__ void mediumKernel(int *offset, int *col_id, int *medium, int *set, int *color, int currentColor)
{
	int node = medium[(blockIdx.x*blockDim.x+threadIdx.x)/32];
	int node_offset = (blockIdx.x*blockDim.x+threadIdx.x)%32;
	int neighLen = offset[node+1]-offset[node];

	for(int i=node_offset; i<neighLen; i=i+32)
	{
		int item = col_id[offset[node]+i];
		if(item >= node && color[item]==0)
			set[node]=0;
	}
	
	__syncthreads();
	
	if(node_offset == 0){
	if(set[node]==1)
		color[node]=currentColor;
	}
}
