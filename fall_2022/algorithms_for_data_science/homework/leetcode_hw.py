"""
Leetcode Assignment - 5 Easy Problems

Joshua D. Ingram

Wednesday, September 14, 2022
"""

# Problem 1 - 1. Two Sum
# Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
# You can return the answer in any order.

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return []

# Accepted
# Runtime: 3873 ms
# Memory: 15MB

# Problem 2 - 2351. First Letter to Appear Twice
# Given a string s consisting of lowercase English letters, return the first letter to appear twice.

class Solution:
    def repeatedCharacter(self, s: str) -> str:
        seen = []
        for letter in s:
            if letter in seen:
                return letter
            else:
                seen.append(letter)

# Accepted
# Runtime: 49 ms
# Memory: 13.8 MB

# Problem 3 - 387. First Unique Character in a String
# Given a string s, find the first non-repeating character in it and return its index. If it does not exist, return -1.

class Solution:
    def firstUniqChar(self, s: str) -> int:
        
        # create a counter as a dict
        counter = {}
        for letter in s:
            if letter not in counter:
                counter[letter] = 0
            counter[letter] += 1
            
        # check for first 1 count in counter
        for i in range(len(s)):
            if counter[s[i]] == 1:
                return i
        return -1

# Accepted
# Runtime: 300 ms
# Memory: 14.3 MB

# Problem 4 - 9. Palindrome Number
# Given an integer x, return true if x is palindrome integer.
# An integer is a palindrome when it reads the same backward as forward.

class Solution:
    def isPalindrome(self, x: int) -> bool:
        reversed = str(x)[::-1]
        if reversed == str(x):
            return True
        else:
            return False

# Accepted
# Runtime: 138 ms
# Memory: 13.8 MB

# Problem 5 - 66. Plus One
# You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. 
# The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.
# Increment the large integer by one and return the resulting array of digits.

class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        
        for i in range(len(digits)-1,-1,-1):
            if digits[i] == 9:
                digits[i] = 0
                if i == 0:
                    digits.insert(0,1)
                    return digits
            else:
                digits[i] += 1
                return digits

# Accepted
# Runtime: 62 ms
# Memory: 13.8 MB