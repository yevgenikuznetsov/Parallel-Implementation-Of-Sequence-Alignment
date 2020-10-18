# Parallel-Implementation-Of-Sequence-Alignment

Each letter in the sequence represents DNA, RNA, or protein. Identification of region of similarity of
set of Sequences is extremely time consuming. This project deals with a simplified version of
Sequence Alignment of two sequences. The purpose of the project is to parallelize the basic algorithm
to produce an efficient computations within MPI, OpenMP and CUDA environment.


## MPI:

The master will manage the Slave processes dynamically and will not participate in calculation. Each Slave will calculate different RNA and will send to the Master their result.
The rational of choosing the specific architecture – When the Slave has finishes calculating the best result, the Master will immediately send him a new RNA for calculation (if there is more in the file). The Master will not participate in calculation, he will have to always listen, summarize, receive and send new Job.
Complexity evaluation – O(NumOfRna / NumOfSlaves)


## CUDA:

Compare two letters (one letter from DNA and the other letter from RNA). We calculate here which group these two letters belong.
The rational of choosing the specific architecture – Cuda can handle massive amount of small task on parallel, In this case, it handle massive number of letter witch we need to compare and calculate their group – it's a perfect mach. 
Gpu have more then 500,000 threads. Witch means that each thread can handle a single letter from DNA and RNA.
Complexity evaluation – O(k) when k is the dimension.


## OMP:

Calculates the best result for specific RNA. Goes over the array that was received from Cuda.
The array contains information for each pair of letters (one letter from DNA and the second from RNA) to which group they belong. With this array we calculate the score and according to this make a comparison to find the maximum score.
The rational of choosing the specific architecture – OMP uses the computer cores to create threads. We can calculate with this threads the score from array – in a single parallel loop.
Complexity evaluation – O(n / numOfThreads)



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Total Complexity:

    O((num of RNA)/(num of Slave)*(N-M)*M)




