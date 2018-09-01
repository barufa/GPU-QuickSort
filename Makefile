run: clean compile
	@clear
	@echo "Ejecutando programas..."
	@echo "Secuencial"
	@./Sort_secuencial
	@echo "Paralelo"
	@./Sort_paralelo
	@echo "GPU-Quicksort"
	@./GPU-Quicksort

compile:
	@clear
	@echo "Compilando..."
	@gcc -Wall -o Sort_secuencial Sort_secuencial.c -lm -g -fopenmp	
	@echo "Sort_secuencial creado..."		
	@gcc -Wall -o Sort_paralelo Sort_paralelo.c -lm -g -fopenmp
	@echo "Sort_paralelo creado..."		
	@nvcc -arch=sm_35 -rdc=true -lcudadevrt -I /usr/local/cuda-9.2/samples/common/inc -o Sort_cuda_simple Sort_cuda_simple.cu
	@echo "Sort_cuda_simple creado..."	
	@nvcc -o GPU-Quicksort GPU-Quicksort.cu
	@echo "GPU-Quicksort creado..."		
	@echo "Compilacion completa."	

clean:
	@clear
	@rm Sort_secuencial
	@rm Sort_paralelo
	@rm Sort_cuda_simple
	@rm GPU-Quicksort
	@echo "Archivos eliminados."
