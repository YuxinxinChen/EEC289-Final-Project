__global__ void smallKernel(int *offset, int *col_id, int *small, int *set, int *color, int currentColor)
{
	int node = small[blockIdx.x*blockDim.x+threadIdx.x];
	int neighLen = offset[node+1]-offset[node];

	for(int i=0; i<neighLen; i++)
	{
		int item = col_id[offset[node]+i];
		if(item >= node && color[item]==0)
		{
			set[node]=0;
			break;
		}

	}

	__syncthreads();
	
	if(set[node]==1)
		color[node]=currentColor;

}
	
