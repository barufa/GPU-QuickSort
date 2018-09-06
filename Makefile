run: clean compile
	@clear
	@echo "Ejecutando programas..."
	@echo "Secuencial"
	@./Quicksort
	@echo "Posix Threads"
	@./Quicksort_Pthreads
	@echo "OpenMP"
	@./Quicksort_OpenMP
	@echo "GPU-Quicksort"
	@./GPU-Quicksort

compile:
	@clear
	@echo "Compilando..."
	@gcc -Wall -o Quicksort Quicksort.c -lm -g -fopenmp	
	@echo "Secuencial creado..."		
	@gcc -Wall -o Quicksort_Pthreads Quicksort_Pthreads.c -lm -g -fopenmp
	@echo "Pthreads creado..."	
	@gcc -Wall -o Quicksort_OpenMP Quicksort_OpenMP.c -lm -g -fopenmp
	@echo "OpenMP creado..."	
	@nvcc -o GPU-Quicksort GPU-Quicksort.cu
	@echo "GPU-Quicksort creado..."		
	@echo "Compilacion completa."	

clean:
	@clear
	@rm Quicksort
	@rm Quicksort_Pthreads
	@rm Quicksort_OpenMP
	@rm GPU-Quicksort
	@echo "Archivos eliminados."
