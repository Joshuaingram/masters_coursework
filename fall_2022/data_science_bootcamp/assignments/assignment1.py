""" NCF Introduction to Data Science in Python Bootcamp
    Assignment 1

    Joshua D. Ingram

    Tuesday, August 23, 2022
"""

class Polygon():

    def __init__(self):
        self.name = "Polygon"

    def get_area(self):
        raise NotImplementedError

    def say_name(self):
        return "Hi, I am a " + self.name

class Square(Polygon):

    def __init__(self, side_length):
        self.name = "Square"
        self.side_length = side_length

    def get_area(self):
        return self.side_length ** 2

"""
TODO:
Implement another subclass or two of Polygon.
At a minimum, you should provide a function implementation
that overrides the get_area function.
"""

# Your implementation here
class Right_triangle(Polygon):

    def __init__(self, base, height):
        self.name = "Right Triangle"
        self.base = base
        self.height = height

    def get_area(self):
        return 0.5 * self.base * self.height

class Rectangle(Polygon):

    def __init__(self, length, width):
        self.name = "Rectangle"
        self.length = length
        self.width = width

    def get_area(self):
        return self.length * self.width
    


"""
For the remaining tasks, complete the function implementations
adhering to the provided specifications.
"""


def all_factors(n):
    """ Return the set of factors of n (including 1 and n).
    You may assume n is a positive integer. Do this in one line for extra credit.

    Example:
    >>> all_factors(24)
    {1, 2, 3, 4, 6, 8, 12, 24}
    >>> all_factors(5)
    {1, 5}
    """
    factors = set()

    for i in range(1,n+1):
        if n % i == 0:
            factors.add(i)

    return factors


def get_student_avg(gradebook_dict, student):
    """ Given a dictionary where each key-value pair is of the form: (student_name, [scores]),
    return the average score of the given student. If the given student does not exist, return -1

    Example:
    >>> get_student_avg({"Sally":[80, 90, 100], "Harry": [75, 80, 85]}, "Sally")
    90.0
    >>> get_student_avg({"Sally":[80, 90, 100], "Harry": [75, 80, 85]}, "John")
    -1
    """
    
    if student in gradebook_dict:
        avg = sum(gradebook_dict[student]) / len(gradebook_dict[student])
        return avg
    else:
        return -1


def every_other(seq):
    """ Returns a new sequence containing every other element of the input sequence, starting with
    the first. If the input sequence is empty, a new empty sequence of the same type should be
    returned.

    Example: every_other("abcde")
    "ace"
    """
    return seq[0:len(seq):2]
            


def all_but_last(seq):
    """ Returns a new sequence containing all but the last element of the input sequence.
    If the input sequence is empty, a new empty sequence of the same type should be returned.

    Example:
    >>> all_but_last("abcde")
    "abcd"
    """
    return seq[0:len(seq)-1]


def substrings(seq):
    """ Returns a set of all the substrings of s.
    Recall we can compute a substring using s[i:j] where 0 <= i, j < len(s).

    Example:
    >>> substrings("abc")
    {"", "a", "ab", "abc", "b", "bc", "c"}
    """
    subs = set()
    for i in range(len(seq)):
        for j in range(i, len(seq)):
            subs.add(seq[i:j+1])
    return subs


def many_any(lst, k):
    """ Returns True if at least k elements of lst are True;
    otherwise False. Do this in one line for extra credit.
    Hint: use a list comprehension.

    Example:
    >>> many_any([True, True, False, True], 3)
    True
    >>> many_any([True, True, False, False], 3)
    False
    """

    if lst.count(True) >= k:
        return True
    else:
        return False

    


def alphabet_construct(seq, alphabet):
    """ Returns True if string s can be constructed from the set of length-1 strings
    alphabet and False otherwise.

    Example:
    >>> alphabet_construct("hello", {"a", "b", "h", "e", "l", "o"})
    True
    >>> alphabet_construct("hello", {"a", "b", "h", "e", "o"})
    False
    """
    for i in range(len(seq)):
        if seq[i] not in alphabet:
            return False
    return True

    # {set(seq).intersection(alphabet) == set(seq) and len(seq) < len(alphabet)}
            



def common_chars(seq, k):
    """ Returns the set of characters that appear more than k times in
    the input string, along with their number of occurrences.

    Example:
    >>> common_chars("cat in a hat", 2)
    {("a", 3), (" ", 3)} 
    """
    # this only works because sets cannot have duplicates. If it were a list, in the example above there would be 3 instances of ("a", 3).
    return {(c, seq(c)) for c in seq if seq.count(c) > k}
    


def dict_to_tuple_list(my_dict):
    """ Given a dictionary where each k-v pair is of the form (x, [y]), convert the dictionary
    into a list of tuples.

    Example:
    >>> dict_to_tuple_list({'x': [1, 2, 3], 'y':[4, 5, 6]})
    [(x, 1), (x, 2), (x, 3), (y, 4), (y, 5), (y, 6)]
    """
    return [(k,v) for k in my_dict for v in my_dict[k]]

