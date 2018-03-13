__global__ void smallKernel(int *offset, int *col_id, int *small, int sizeSmall, int *color, int currentColor)
{
	if((blockIdx.x*blockDim.x+threadIdx.x)<sizeSmall)
	{
	    int node = small[blockIdx.x*blockDim.x+threadIdx.x];

	    if(color[node]==0) {
	        int neighLen = offset[node+1]-offset[node];
	        bool set = 1;
	        for(int i=0; i<neighLen; i++)
	        {
		   int item = col_id[offset[node]+i];
		   if(item >= node && (color[item]==0 || color[item]==currentColor))
		   {
			set = 0;
			break;
		   }
	        } 
 	        __syncthreads();
	
	        if(set==1)
		   color[node]=currentColor;
	   }
       }
}
	
