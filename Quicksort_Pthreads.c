#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <unistd.h>
#define N 30000000
#define M 50000

#if N<100000//Si la cantidad de numeros a ordenar es menor a 100000, el algoritmo se comporta como si fuera secuencial
#define H 0
#else
#define H (int)get_nprocs()//Caso contrario lanzo tantos hilos como nucleos tenga el procesador
#endif

typedef struct {
    int *v,b,t,c;
}qsparams;

qsparams mkqsparams(int *v,int b,int t,int c){
	qsparams r;
	r.v = v;
	r.b = b;
	r.t = t;
	r.c = c;
	return r;
}

int Quicksortaux(int *v, int b, int t){
    
    int i,temp;
    int pivote, valor_pivote;

    pivote = b;
    valor_pivote = v[pivote];
    for (i=b+1; i<=t; i++){
        if (v[i] < valor_pivote){
            pivote++;
			temp=v[i];
			v[i]=v[pivote];
			v[pivote]=temp;
        }
    }
    temp=v[b];
    v[b]=v[pivote];
    v[pivote]=temp;
    return pivote;
}

void *Quicksort(void *p){
    
    qsparams *params = (qsparams *)p;//Desglozo los parametros en variables
    int *v = params->v;
    int b = params->b;
    int t = params->t;
    int c = params->c;
    int pivote;
    if(b < t){
        pivote=Quicksortaux(v, b, t);//Ordeno parcialmente el arreglo de manera secuencial
        qsparams params1 = mkqsparams(v,b,pivote-1,c*2);
        qsparams params2 = mkqsparams(v,pivote+1,t,c*2+1);
        if(c<H && (pivote-1-b)>M){
            pthread_t t1;//Lanzo un nuevo hilo para la primer parte del arreglo
            pthread_create(&t1,0,Quicksort,(void *)&params1);
            Quicksort((void *)&params2);
            pthread_join(t1, NULL);
        }else{
            Quicksort((void *)&params1);
            Quicksort((void *)&params2);
        }
    }
    
    return NULL;
}

int main(void){
    
    int *a,i;
    srand(time(0)^(getpid()));
    a = (int*)malloc(N*sizeof(int));
    for(i=0;i<N;i++)a[i]=random()%N+1;
    qsparams params = mkqsparams(a,0,N-1,1);//Inicializo los parametros de la funcion
    
    double times=omp_get_wtime();//Guardo el tiempo en el que empieza a ejecutarse el algoritmo
    Quicksort((void *)&params);
    times = omp_get_wtime()-times;//Calculo la diferencia
    printf("Tiempo : %f ms.\n",times*1000);
    
    free(a);
    
    return 0;
}
