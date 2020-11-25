import unittest

from blindml.frontend.config.task.task import parse_task_capsule


class MyTestCase(unittest.TestCase):
    def test_task_capsule_parsing(self):
        task = parse_task_capsule("perovskite_task.jsonnet")

    def test_model_search(self):
        task = parse_task_capsule("perovskite_task.jsonnet")
        task.search_for_model()


if __name__ == "__main__":
    unittest.main()
