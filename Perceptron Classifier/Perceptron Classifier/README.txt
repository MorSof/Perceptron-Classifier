# Perceptron-Classifier
Given a set of N points in K-dimensional space.
Each point X is marked as belonging to set A or B.
Implement a Simplified Binary Classification algorithm to find a Linear Classifier.
The result depends on the maximum iteration allowed, value of the chosen parameter a and the time value t.
The purpose of the project is to define a minimal value of t that leads to the Classifier with acceptable value of Quality of Classifier.

IN THIS PROJECT I HAVE TWO DIFFERENT SOLUTIONS (uncomment the relevant line in the binaryClassificationAlgorithm function)

# First Solution


#MPI usage:

The Master process will manage the Slave processes dynamically and will not participate in the slave's "dirty" work.
Each Slave will calculate different time values with the appropriate time difference and will send to the master their results according to if q<QC.
The Master will choose the minimal successful time value among all the times that were been calculated.
If all the slaves failed, the master will tell them to continue to search in their next time interval.
If only the master will be activated without other slaves, he will do all the work alone.
The rational of choosing the specific architecture - 
In case that the program will not find the correct solution in the first time interval, it will have the next time solution right away,
depends on the number of processes which were activeted.
The master will not participate in the perceptron algorithem in case there is large amount of slaves, he will have to always listen, 
summerize, recieve and send - He needs to be the "quick manager".
complexity evaluation - 
(tmax/dt) / numOfSlaves)


#Cuda Usage:

Will calculate and set the new points coordinates according to the local time the slave is handle.
The rational of choosing the specific architecture - 
The big advantage of Cuda is that it can handle massive amount of small tasks on parallel, 
In this case, it handle massive amount of points which need to be relcataed - its a perfect match!
complexity evaluation - 
In this exrecise the max amount of input points are 500,000, and Invidia GPU have more then 500,000 threads. 
Which means that Each Cuda thread can handle a single point, loop throgh its demensions - O(k) in parallel - O(k) total.


#OMP usage:

Will check if each point is in the right position relative to the linear line and will stop the loop if all the points are in their right place.
will count the number of points that are not in their dedicated position (Nmiss), and will remember the minimum index of the point that was not in the right position. 
The rational of choosing the specific architecture - 
OMP uses the computer cores to create threads. I used it here because it have the reduction action to sum all the Nmiss,
and reduction to find the minimum index of a points that was not in the right position - all in a single parallel loop!
complexity evaluation - 
One iteration: O((N/numOfThreads)K)
LIMIT iterations: O((N / numOfThreads) * K * LIMIT)


Total Complexity: O( (O(k) + O((N / numOfThreads) * K * LIMIT)) * ((tmax/dt) / numOfSlaves) )


# Second Solution


#MPI usage:
(stay the same as the original solution)


#Cuda Usage:

Two useges:

1. Same usage as the first solution.
2. Inside the LIMIT loop - 
Will check if each point is in the right position relative to the linear line and will put the results in an array. 
will stop the loop if all the points are in the right place.
complexity evaluation - 
In this exrecise the max amount of input points are 500,000, and Invidia GPU have more then 500,000 threads. 
Which means that Each Cuda thread can handle a single point, loop throgh its demensions - O(k) + O(k) in parallel - O(k) total.

#OMP usage:

Inside the LIMIT loop - 
will count the number of points that are not in their dedicated position according to the result array that the Cuda has been calculated.
In addition, will remember the minimum index of the point that was not in the right position. 
complexity evaluation - 
O(n/numOfThreads)


Total Complexity: O( (O(k) + O((N / numOfThreads) * LIMIT)) * ((tmax/dt) / numOfSlaves) )
