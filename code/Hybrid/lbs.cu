__global__
void WorkItemRank(int *scan, int *lbs, int *wir, int sizeLbs) {
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeLbs; i+=blockDim.x*gridDim.x)
	{
		wir[i] = i - scan[lbs[i]];
	}
}

__global__
void FindChangeColor(int sizeLarge, int *nodes, int *wir, int *lbs, int sizeLbs, int *col_id, int *offset, int currentColor, int *color, int *set) {
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeLbs; i+=blockDim.x*gridDim.x)
	{
		int neighborOwner = lbs[i];
		int neighbor = col_id[offset[nodes[lbs[i]]] + wir[i]];

		if(neighbor >= neighborOwner && (color[neighbor]==0 || color[neighbor]==currentColor))
			set[neighborOwner]=0;
	
	}
}

__global__
void assignColor(int *set, int sizeLarge, int *Large, int *color, int currentColor)
{
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeLarge; i=i+gridDim.x*blockDim.x)
	{
		if(set[i]==1 && color[Large[i]]==0)
			color[Large[i]]=currentColor;
		set[i]=1;
	}
}
