#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <time.h>
#include <unistd.h>
#define MAXR(sz) (((sz)+MAXSEQ-1)/MAXSEQ+1)
#define MAXT MAXR(MAXN)
int MAXN;
int MAXSEQ;
int THRN;

//===Definicion de estructuras y funciones utiles===

typedef struct secuence{
	int start,end,pivot;
}secuence;

typedef struct block{
	secuence seq,parent;
	int blockcount,id,bid;
}block;

__host__ __device__ secuence mkseq(int s,int e,int p){
	secuence r;

	r.start=s;
	r.end=e;
	r.pivot=p;

	return r;
}

__host__ __device__ block mkblock(secuence s,secuence p,int b,int id,int bid){

	block r;

	r.seq=s;
	r.parent=p;
	r.blockcount=b;
	r.id=id;
	r.bid = bid;

	return r;
}
__host__ __device__ int MED(int x,int y,int z){

	int m=min(min(x,y),z),M=max(max(x,y),z);

	if(m<x && x<M)return x;
	if(m<y && y<M)return y;
	return z;
}

//===Implementacion del algoritmo GPU-Quicksort===

int *LT,*GT;

__global__ void gqsort1(block * blocks,int * d,int * LT,int * GT){
	
	int id = blockIdx.x,th = threadIdx.x,cth = blockDim.x;
	int gt=0,lt=0,pivot=blocks[id].seq.pivot;
	int start = blocks[id].seq.start,end = blocks[id].seq.end;
	
	if(th==0){
		LT[id]=0;
		GT[id]=0;
	}
	__syncthreads();
	
	for(int j=start+th;j<end;j+=cth){
		if(d[j]<pivot)lt++;
		else if(d[j]>pivot)gt++;
	}
	
	atomicAdd(&LT[id],lt);
	atomicAdd(&GT[id],gt);
	
	return;
}

__global__ void gqsort2(block * blocks,int * d,int * _d,secuence * result,int * LT,int * GT){
	
	int id = blockIdx.x,th = threadIdx.x,cth = blockDim.x;
	__shared__ int lt,gt,lfrom,gfrom;
	
	int start = blocks[id].seq.start, end = blocks[id].seq.end, pivot = blocks[id].seq.pivot;
	int desde = blocks[id].parent.start, hasta = blocks[id].parent.end;
	int llt=0,lgt=0;
	
	
	if(th==0){
		lt=gt=0;
		lfrom = desde;
		gfrom = hasta;
	}
	__syncthreads();
	
	for(int j=id-blocks[id].id+th;j<id+1;j+=cth)lgt+=GT[j];
	for(int j=id-blocks[id].id+th;j<id;j+=cth)llt+=LT[j];
	
	atomicAdd(&lt,llt);
	atomicAdd(&gt,lgt);
	atomicAdd(&lfrom,llt);
	atomicSub(&gfrom,lgt);
	__syncthreads();
	
	for(int j=th;j<end-start;j+=cth){
		if(d[j+start]<pivot){
			int old = atomicAdd(&lfrom,1);
			_d[old]=d[j+start];
		}else if(d[j+start]>pivot){
			int old = atomicAdd(&gfrom,1);
			_d[old]=d[j+start];
		}
	}
	if(blocks[id].id==blocks[id].blockcount-1){
		int lstart = blocks[id].parent.start,lend=blocks[id].parent.start+lt+LT[id];
		int gstart = blocks[id].parent.end-gt, gend = blocks[id].parent.end;
		
		for(int j=lend+th;j<gstart;j+=cth)_d[j]=pivot;
		
		if(th==0){
			result[blocks[id].bid*2] = mkseq(lstart,lend,MED(_d[lstart],_d[(lstart+lend)/2],_d[lend-1]));
			result[blocks[id].bid*2+1] = mkseq(gstart,gend,MED(_d[gstart],_d[(gstart+gend)/2],_d[gend-1]));
		}
	}
	
	return;
}

__global__ void gqsort3(block * blocks,int * d,int * _d){

	int id = blockIdx.x,th = threadIdx.x,cth = blockDim.x;
	int start = blocks[id].seq.start,end = blocks[id].seq.end;
	for(int j=start+th;j<end;j+=cth)
		d[j] = _d[j];
	
	return;
}


void gqsort(thrust::host_vector<block> & blocks,int * d,int * _d,secuence * result){
	
	thrust::device_vector<block> blocks_dev(blocks.size());
	
	thrust::copy(blocks.begin(),blocks.end(),blocks_dev.begin());
	block * pblocks = thrust::raw_pointer_cast(&blocks_dev[0]);
	int size = blocks_dev.size();
	
	gqsort1<<<size,THRN>>>(pblocks,d,LT,GT);
	gqsort2<<<size,THRN>>>(pblocks,d,_d,result,LT,GT);
	gqsort3<<<size,THRN>>>(pblocks,d,_d);
	
	return;
}

void lqsort(thrust::host_vector<secuence> & done,thrust::device_vector<int> & d){
	for(int i=0;i<done.size();i++)if(done[i].end-done[i].start>1){
		if(done[i].start<done[i].end)
			thrust::sort(d.begin()+done[i].start,d.begin()+done[i].end);
	}
	return;
}

void gpuqsort(thrust::host_vector<int> & v){
	
	int size = v.size(),startpivot = MED(v[0],v[size/2],v[size-1]);
	
	thrust::host_vector<block> blocks;
	thrust::host_vector<secuence> done,work;
	thrust::device_vector<int> d(size),_d(size);
	
	int *pd = thrust::raw_pointer_cast(&d[0]),*_pd = thrust::raw_pointer_cast(&_d[0]);
	secuence *result,*news;
	
	cudaMalloc(&LT,MAXT*sizeof(int));
	cudaMalloc(&GT,MAXT*sizeof(int));
	cudaMalloc(&result,MAXR(size)*sizeof(secuence));
	cudaMallocHost(&news,MAXR(size)*sizeof(secuence));
	thrust::copy(v.begin(),v.end(),d.begin());	
	work.push_back(mkseq(0,d.size(),startpivot));
	
	while(!work.empty() && work.size()+done.size()<MAXSEQ && MAXSEQ<size){
		int blocksize = 0,rsz = work.size()*2;
		for(int i=0;i<work.size();i++)
			blocksize += (work[i].end-work[i].start)/MAXSEQ;
		blocks.clear();
		for(int i=0;i<work.size();i++){
			int start = work[i].start, end = work[i].end, pivot = work[i].pivot;
			int blockcount = (end-start+blocksize-1)/blocksize;
			secuence parent = work[i];
			for(int j=0;j<blockcount-1;j++){
				int bstart = start + blocksize*j;
				blocks.push_back(mkblock(mkseq(bstart,bstart+blocksize,pivot),parent,blockcount,j,i));
			}
			blocks.push_back(mkblock(mkseq(start+blocksize*(blockcount-1),end,pivot),parent,blockcount,blockcount-1,i));
		}
		gqsort(blocks,pd,_pd,result);
		work.clear();
		cudaMemcpy(news,result,rsz*sizeof(secuence),cudaMemcpyDeviceToHost);
		for(int i=0;i<rsz;i++){
			if(news[i].end-news[i].start<size/MAXSEQ)
				done.push_back(news[i]);
			else
				work.push_back(news[i]);
		}
	}
	for(int i=0;i<work.size();i++)done.push_back(work[i]);
	lqsort(done,d);
	thrust::copy(d.begin(),d.end(),v.begin());
	
	d.clear();
	_d.clear();
	cudaFree(LT);
	cudaFree(GT);
	cudaFree(result);
	
	return;
}

int main(int argc, char *argv[]){
	
	if(argc!=4)return 0;
	MAXN = atoi(argv[1]);
	MAXSEQ = atoi(argv[2]);
	THRN = atoi(argv[3]);
	srand(time(0)^(getpid()));
	
	thrust::host_vector<int> v;
	
	srand(time(0)^(getpid()));
	for(int i=0;i<MAXN;i++){
		v.push_back(rand()%MAXN+1);
	}
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	gpuqsort(v);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%.3f\n",milliseconds);
	
	v.clear();
	
	return 0;
}

