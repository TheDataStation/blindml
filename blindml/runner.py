import time

from blindml.frontend.config.task.task import parse_task_capsule


def run(task_file_fp):
    task = parse_task_capsule(task_file_fp)
    task.search_for_model()
    while not task.get_model_search_update():
        print("no model trained yet")
        time.sleep(5)

    model = task.train_best_model()
    task.get_explanations(model)


def run_wit(task_file_fp):
    task = parse_task_capsule(task_file_fp)
    print(task)
    task.search_for_model()
    while not task.get_model_search_update():
        print("no model trained yet")
        time.sleep(5)

    model = task.train_best_model()
    task.get_wit(model)


if __name__ == "__main__":
    # main("/Users/maksim/dev_projects/blindml/tests/logan_task.jsonnet")
    # main("/tests/logan_task.jsonnet")
    run("/Users/maksim/dev_projects/blindml/tests/perovskite_task.jsonnet")
