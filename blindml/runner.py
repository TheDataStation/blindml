from pprint import pprint

from blindml.frontend.config.task.task import parse_task_capsule
from blindml.frontend.reporting.metrics import plot_trial_record


def main(task_file_fp):
    task = parse_task_capsule(task_file_fp)
    task.search_for_model()
    # metric_values = task.get_model_search_update()
    # plot_trial_record(metric_values)
    # res = task.train_best_model()
    # print(res["hyper_parameters"])
    # task.save_model(res, "/Users/maksim/dev_projects/blindml/tests")
    # res = task.load_model("/Users/maksim/dev_projects/blindml/tests")
    # pprint(res["scores"])


if __name__ == "__main__":
    # main("/Users/maksim/dev_projects/blindml/tests/logan_task.jsonnet")
    main("/Users/maksim/dev_projects/blindml/tests/perovskite_task.jsonnet")
    # main("/tests/perovskite_task.jsonnet")
