class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        numbers = [0] *len(nums)
        for i in range(len(nums)):
            numbers[nums[i]] =+ 1
        for i in range(len(numbers)):
            if numbers[i] >= len(nums):
                return i


if __name__=='__main__':
    nums = [1, 2, 3, 2, 2, 2, 5, 4, 2]
    print(Solution().majorityElement(nums))