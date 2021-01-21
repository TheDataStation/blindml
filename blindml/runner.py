import time

from blindml.frontend.config.task.task import parse_task_capsule
from halo import HaloNotebook as Halo

def run(task_file_fp):
    task = parse_task_capsule(task_file_fp)
    task.search_for_model()
    while not task.get_model_search_update():
        print("no model trained yet")
        time.sleep(5)

    model = task.train_best_model()
    task.get_explanations(model)


def run_wit(task_file_fp, min_models=10):
    print("Loading task capsule")
    task = parse_task_capsule(task_file_fp)
    task.search_for_model()
    spinner = Halo(text='Searching model space', spinner='dots')
    spinner.start()
    time.sleep(1)
    n_models = len(task.get_model_search_update())
    if n_models >= min_models:
        for i in range(n_models):
            spinner.text = f"\t{i} models considered"
            time.sleep(5/n_models)
    else:
        while n_models < min_models:
            n_models = len(task.get_model_search_update())
            spinner.text = f"\t{n_models} models considered"
            time.sleep(.1)
    spinner.succeed(f"{len(task.get_model_search_update())} models considered")
    print("Training best model")
    model = task.train_best_model()
    task.get_wit(model)


if __name__ == "__main__":
    # main("/Users/maksim/dev_projects/blindml/tests/logan_task.jsonnet")
    # main("/tests/logan_task.jsonnet")
    run("/Users/maksim/dev_projects/blindml/tests/perovskite_task.jsonnet")
