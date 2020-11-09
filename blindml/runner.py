import time
from pprint import pprint

from gooey import GooeyParser

# from nni_cmd.nnictl import parse_args
from nni_cmd.nnictl_utils import stop_experiment

from blindml.frontend.config.data.statistics import get_data_stats
from blindml.frontend.config.task.parser import parse_task_capsule
from blindml.nni_helper import NniArgs, create_experiment, trial_ls, list_experiment

hours = 1

experiment_config = {
    "authorName": "default",
    "experimentName": "maxs_experiment",
    "maxExecDuration": f"{hours * 60 * 60}",  # s
    "maxTrialNum": 1000,
    "searchSpacePath": "/Users/maksim/dev_projects/blindml/blindml/backend/search/search_space.json",
    "trainingServicePlatform": "local",
    "trial": {
        "codeDir": "/Users/maksim/dev_projects/blindml/",
        "command": "python3 -m blindml.backend.run",
        "gpuNum": 0,
    },
    "trialConcurrency": 1,
    "tuner": {"builtinTunerName": "TPE", "classArgs": {"optimize_mode": "minimize"}},
    "useAnnotation": False,
}


def run_nni():
    stop_experiment(NniArgs(id="maxs_experiment", all=True))
    time.sleep(5)
    experiment_id = "maxs_experiment"
    create_experiment(
        NniArgs(foreground=False, debug=False, port=8080), experiment_config
    )
    # parse_args()


def get_experiment_update(experiment_id):
    list_experiment(NniArgs(id=experiment_id))
    trials = trial_ls(NniArgs(id=experiment_id, head=False, tail=False))
    good_trials = [t for t in trials if "finalMetricData" in t]
    top_trial = sorted(
        good_trials,
        key=lambda t: float(t["finalMetricData"][0]["data"].replace('"', ""))
        if experiment_config["tuner"]["classArgs"]["optimize_mode"] == "minimize"
        else -float(t["finalMetricData"][0]["data"].replace('"', "")),
    )
    pprint(top_trial[0])


# @Gooey(target="ffmpeg", program_name="Frame Extraction v1.0", suppress_gooey_flag=True)
def main():
    parser = GooeyParser(description="BlindML")
    parser.add_argument("-f", help="Task capsule file path", dest="task_file_fp")
    args = parser.parse_args()

    df, y_col = parse_task_capsule(args.task_file_fp)
    _, corr = get_data_stats(df)
    # show_correlation(corr)
    run_nni()


if __name__ == "__main__":
    run_nni()
