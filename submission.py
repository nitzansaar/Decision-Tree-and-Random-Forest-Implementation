from unittest import TestCase


from sklearn_intro import ZeroR, RandR


class Test(TestCase):
    def test_ZeroR(self):
        data = [1, 1, 2]
        result = ZeroR(data)
        self.assertEqual(result, 1)

    def test_RandR(self):
        data = ['cat', 'cat', 'cat', 'dog']
        results = {'cat': 0, 'dog': 0}
        for i in range(100):
            result = RandR(data)
            results[result] += 1

        print(results)

