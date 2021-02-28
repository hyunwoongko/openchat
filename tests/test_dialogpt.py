import unittest
from dialogpt_chat import DialoGPT


class TestDialoGPT(unittest.TestCase):

    def test_run(self):
        gpt = DialoGPT('large', 'cpu')
        gpt.run()

    def test_single_turn(self):
        gpt = DialoGPT('large', 'cpu')
        _out = gpt.predict("user_id", "Hello.")
        print(_out)

    def test_multi_turn(self):
        gpt = DialoGPT('large', 'cpu')

        while True:
            _in = input("user : ")
            _out = gpt.predict(user_id="user_id", text=_in)
            print(f"bot : {_out}")


if __name__ == '__main__':
    testcase = TestDialoGPT()
    testcase.test_run()
