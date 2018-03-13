//each kernel process one node
__global__ void largeKernel(int *offset, int *col_id, int *large, int sizeLarge, int *color, int currentColor)
{
	__shared__ bool set[1];
	//get the node from large array
	if(blockIdx.x < sizeLarge)
	{
	    set[0]=1;
	    int node = large[blockIdx.x];
	    if(color[node]==0)
	    {
	    	int neighLen = offset[node+1]-offset[node];

	    	for(int i = threadIdx.x; i<neighLen; i=i+blockDim.x)
	    	{
		   int item = col_id[offset[node]+i];
		   if(item >= node && color[item]==0)
			set[0]=0;
	      	}
  	    	__syncthreads();

	        if(threadIdx.x == 0){
	   		if(set[0] == 1)
				color[node]=currentColor;
		}
	    }
	}
}

	
