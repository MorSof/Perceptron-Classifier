# Perceptron-Classifier
Given a set of N points in K-dimensional space.
Each point X is marked as belonging to set A or B.
Implement a Simplified Binary Classification algorithm to find a Linear Classifier.
The result depends on the maximum iteration allowed, value of the chosen parameter a and the time value t.
The purpose of the project is to define a minimal value of t that leads to the Classifier with acceptable value of Quality of Classifier.

IN THIS PROJECT I HAVE TWO DIFFERENT SOLUTIONS:

####################First Solution############################

#MPI usage:

The Master process will manage the Slave processes dynamically and will not participate in the slave's "dirty" work.
Each Slave will calculate different time values with the appropriate time difference and will send to the master their results according to if q<QC.
The Master will choose the minimal successful time value among all the times that were been calculated.
If all the slaves failed, the master will tell them to continue to search in their next time interval.
If only the master will be activated without other slaves, he will do all the work alone.
Why I chose this way:
In case that the program will not find the correct solution in the first time interval, it will have the next time solution right away.


#Cuda Usage:

Will calculate and set the new points coordinates according to the local time the slave is handle.


#OMP usage:

Inside the LIMIT loop:
Will check if each point is in the right position relative to the linear line and will stop the loop if all the points are in their right place.
will count the number of points that are not in their dedicated position, and will remember the minimum index of the point that was not in the right position. 


#################Second Solution:######################


#MPI usage:

(stay the same as the original solution)


#Cuda Usage:

1. Will calculate and set the new points coordinates according to the local time the slave is handle.
2. Inside the LIMIT loop - 
Will check if each point is in the right position relative to the linear line and will put the results in an array. 
will stop the loop if all the points are in the right place.


#OMP usage:

Inside the LIMIT loop - 
will count the number of points that are not in their dedicated position according to the result array that the Cuda has been calculated.
In addition, will remember the minimum index of the point that was not in the right position. 
