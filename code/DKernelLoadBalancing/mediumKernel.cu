//numThreads should be multiple of 32
__global__ void mediumKernel(int *offset, int *col_id, int *medium, int sizeMedium, int *color, int currentColor)
{
	extern __shared__ bool set[];
	if( (blockIdx.x*blockDim.x+threadIdx.x)/32 < sizeMedium) {
	    int node = medium[(blockIdx.x*blockDim.x+threadIdx.x)/32];
	    if(color[node]==0) {
		int node_offset = (blockIdx.x*blockDim.x+threadIdx.x)%32;
		int neighLen = offset[node+1]-offset[node];

		set[threadIdx.x/32]=1;
		__syncthreads();
		for(int i=node_offset; i<neighLen; i=i+32)
		{
		    int item = col_id[offset[node]+i];
		    if(item >= node && (color[item]==0 || color[item]==currentColor))
			set[threadIdx.x/32]=0;
		}
	
		__syncthreads();
	
		if(node_offset == 0){
		   if(set[threadIdx.x/32]==1)
		   	color[node]=currentColor;
		}
	     }
	}
}
