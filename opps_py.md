## oops in python
* class: it is an blueprint how the object behave.
* method vs function
    * method are functions that is written inside a class. like L is List L.append(1)
    * funtions are normal function that is not written inside the class and it is available for all. like len(L)

## constructor
* created using always __init__
* Here we can say __init__ is constructor that intializes object
* def __init__(self): here self is an object. THe object you are workin currently that is self

* why inside class every methods needed self?
    * In Python, self refers to the instance of the class. It's used inside class methods to access instance variables and other methods. When we call a method on an object, Python automatically passes the object itself as the first argument — and we use self to receive and use it within the method. It helps Python differentiate between instance variables and local variables.

* extra notes
    * class ke objects are also mutable like list ,dict and sets