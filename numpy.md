*  Numpy
    - numpy is a python library used to perform statistical operations.

* Numpy array vs python list
    - numpy array is fixed size but python list grow dynamically
    - numpy array is used to store same type of data type and python array can store different data types.
* creating numpy array
     ```python
     import numpy as np
     a=np.array([1,2,3])

     # 2d
     np.array([[1,2,3],[4,5,6]])

     # dtype(it can make any data type of data)
     np.array([1,2,3],dtype=float) # it generates the array of float

     # np.arrang(use to make the array in range)
     np.array(1,11) # output of that is [1,2,3,4,5,6,7,8,9,10]

     #np.reshape(it use to make the 1d to 2d)
     np.arrange(1,13).reshape(2,6) =>it makes the 1d array into two rows and 6 columns

     #np.ones(all the value of array is one)
     np.ones(5)-> makes 1d array with 5 ones
     np.ones((2,3))->make 2d array 2*3

     #np.zeros(same as ones)

     #np.random
     np.random.random((3,4))-> makes the 3*4 matrix of random values between 0 to 1

     #np.linespace(it generates the array in range)
     np.linspace(-10,7,10)->it generates the array of size 10 containg value from -10 to 7 at same size difference

     #np.identity
     it creates the identity matrix


* Array atribues
    ```python
    #ndim(used to find dimension like 1d,2d,3d)
    a1.ndim

    # shape(tells the shape)
    a1.shape

    #size(tells the no of element in that)
    a1.size

    # dtype
    a1.dtpye

    #itemsize

* changing datatypes
    - astype is used to change the data type like a1.astype(np.int32)


* scaler operations
    ```python
    #Arithmatic
    a1*2
    a1+2
    a1**2

    #relational
# Array functions
    ```python
    # max/min/sum/prod
    np.sum(a1)-> give the sum of a1
    np.sum(a1,axis=0)->give the column wise sum
    np.sum(a1.axis=1)->give the row wise sum

    # mean/median/std/var
    # dot product
    np.dot(a1,a2)

    # log and exp(exponents)
    # round/floor.ceil

# indexing and slicing
    * indexing used to fetch the single element
    * slicing(used to fetch multiple element)
        * like a1[2:5]-> it returns the element of 2 to 4 index
        * in 3d i wnat to find an 1st row then a1[0,:] for third columna1[:,2]

# reshaping
    * reshape
    * Transpose
    * raval- covert the all in 1d array

# stacking
    np.hstac(a1,a2) for horizontal stacking
    np.vstack(a,a2) for vertical stacking

# splitting(jus opposite of stacking)







