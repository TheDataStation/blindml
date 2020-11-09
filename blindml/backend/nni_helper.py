import json
import os
import sys
import tempfile
import time
from functools import cmp_to_key
from subprocess import check_call, CalledProcessError

from nni.package_utils import get_builtin_module_class_name
from nni_annotation import expand_annotations, generate_search_space
from nni_cmd.command_utils import kill_command
from nni_cmd.common_utils import (
    print_error,
    get_user,
    get_json_content,
    print_normal,
    detect_process,
)
from nni_cmd.config_utils import Config, Experiments
from nni_cmd.constants import INSTALLABLE_PACKAGE_META, ERROR_INFO, REST_TIME_OUT
from nni_cmd.launcher import (
    get_log_path,
    print_log_content,
    start_rest_server,
    set_platform_config,
    set_experiment,
)
from nni_cmd.nnictl_utils import (
    get_config_filename,
    convert_time_stamp_to_date,
    stop_experiment,
)
from nni_cmd.rest_utils import (
    check_rest_server,
    check_rest_server_quick,
    rest_get,
    check_response,
)
from nni_cmd.url_utils import get_local_urls, trial_jobs_url, experiment_url


# from nni_cmd.nnictl import parse_args


class NniArgs(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def launch_experiment(
    args, experiment_config, mode, config_file_name, experiment_id=None
):
    """follow steps to start rest server and start experiment"""
    nni_config = Config(config_file_name)
    # check packages for tuner
    package_name, module_name = None, None
    if experiment_config.get("tuner") and experiment_config["tuner"].get(
        "builtinTunerName"
    ):
        package_name = experiment_config["tuner"]["builtinTunerName"]
        module_name, _ = get_builtin_module_class_name("tuners", package_name)
    elif experiment_config.get("advisor") and experiment_config["advisor"].get(
        "builtinAdvisorName"
    ):
        package_name = experiment_config["advisor"]["builtinAdvisorName"]
        module_name, _ = get_builtin_module_class_name("advisors", package_name)
    if package_name and module_name:
        try:
            stdout_full_path, stderr_full_path = get_log_path(config_file_name)
            with open(stdout_full_path, "a+") as stdout_file, open(
                stderr_full_path, "a+"
            ) as stderr_file:
                check_call(
                    [sys.executable, "-c", "import %s" % (module_name)],
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
        except CalledProcessError:
            print_error("some errors happen when import package %s." % (package_name))
            print_log_content(config_file_name)
            if package_name in INSTALLABLE_PACKAGE_META:
                print_error(
                    "If %s is not installed, it should be installed through "
                    "'nnictl package install --name %s'" % (package_name, package_name)
                )
            exit(1)
    log_dir = experiment_config["logDir"] if experiment_config.get("logDir") else None
    log_level = (
        experiment_config["logLevel"] if experiment_config.get("logLevel") else None
    )
    # view experiment mode do not need debug function, when view an experiment, there will be no new logs created
    foreground = False
    if mode != "view":
        foreground = args.foreground
        if log_level not in ["trace", "debug"] and (
            args.debug or experiment_config.get("debug") is True
        ):
            log_level = "debug"
    # start rest server
    rest_process, start_time = start_rest_server(
        args.port,
        experiment_config["trainingServicePlatform"],
        mode,
        config_file_name,
        foreground,
        experiment_id,
        log_dir,
        log_level,
    )
    nni_config.set_config("restServerPid", rest_process.pid)
    # Deal with annotation
    if experiment_config.get("useAnnotation"):
        path = os.path.join(tempfile.gettempdir(), get_user(), "nni", "annotation")
        if not os.path.isdir(path):
            os.makedirs(path)
        path = tempfile.mkdtemp(dir=path)
        nas_mode = experiment_config["trial"].get("nasMode", "classic_mode")
        code_dir = expand_annotations(
            experiment_config["trial"]["codeDir"], path, nas_mode=nas_mode
        )
        experiment_config["trial"]["codeDir"] = code_dir
        search_space = generate_search_space(code_dir)
        experiment_config["searchSpace"] = json.dumps(search_space)
        assert search_space, ERROR_INFO % "Generated search space is empty"
    elif experiment_config.get("searchSpacePath"):
        search_space = get_json_content(experiment_config.get("searchSpacePath"))
        experiment_config["searchSpace"] = json.dumps(search_space)
    else:
        experiment_config["searchSpace"] = json.dumps("")

    # check rest server
    running, _ = check_rest_server(args.port)
    if running:
        print_normal("Successfully started Restful server!")
    else:
        print_error("Restful server start failed!")
        print_log_content(config_file_name)
        try:
            kill_command(rest_process.pid)
        except Exception:
            raise Exception(ERROR_INFO % "Rest server stopped!")
        exit(1)
    if mode != "view":
        # set platform configuration
        set_platform_config(
            experiment_config["trainingServicePlatform"],
            experiment_config,
            args.port,
            config_file_name,
            rest_process,
        )

    # start a new experiment
    print_normal("Starting experiment...")
    # set debug configuration
    if mode != "view" and experiment_config.get("debug") is None:
        experiment_config["debug"] = args.debug
    response = set_experiment(experiment_config, mode, args.port, config_file_name)
    if response:
        if experiment_id is None:
            experiment_id = json.loads(response.text).get("experiment_id")
        nni_config.set_config("experimentId", experiment_id)
    else:
        print_error("Start experiment failed!")
        print_log_content(config_file_name)
        try:
            kill_command(rest_process.pid)
        except Exception:
            raise Exception(ERROR_INFO % "Restful server stopped!")
        exit(1)
    if experiment_config.get("nniManagerIp"):
        web_ui_url_list = [
            "{0}:{1}".format(experiment_config["nniManagerIp"], str(args.port))
        ]
    else:
        web_ui_url_list = get_local_urls(args.port)
    nni_config.set_config("webuiUrl", web_ui_url_list)

    # save experiment information
    nnictl_experiment_config = Experiments()
    nnictl_experiment_config.add_experiment(
        experiment_id,
        args.port,
        start_time,
        config_file_name,
        experiment_config["trainingServicePlatform"],
        experiment_config["experimentName"],
    )

    if mode != "view" and args.foreground:
        try:
            while True:
                log_content = rest_process.stdout.readline().strip().decode("utf-8")
                print(log_content)
        except KeyboardInterrupt:
            kill_command(rest_process.pid)
            print_normal("Stopping experiment...")


def create_experiment(args, experiment_config):
    experiment_name = experiment_config.get("experimentName")
    nni_config = Config(experiment_name)
    nni_config.set_config("experimentConfig", experiment_config)
    nni_config.set_config("restServerPort", args.port)
    try:
        launch_experiment(
            args, experiment_config, "new", experiment_name, experiment_name
        )
    except Exception as exception:
        nni_config = Config(experiment_name)
        rest_server_pid = nni_config.get_config("restServerPid")
        if rest_server_pid:
            kill_command(rest_server_pid)
        print_error(exception)
        exit(1)


def trial_ls(args):
    """List trial"""

    def final_metric_data_cmp(lhs, rhs):
        metric_l = json.loads(json.loads(lhs["finalMetricData"][0]["data"]))
        metric_r = json.loads(json.loads(rhs["finalMetricData"][0]["data"]))
        if isinstance(metric_l, float):
            return metric_l - metric_r
        elif isinstance(metric_l, dict):
            return metric_l["default"] - metric_r["default"]
        else:
            print_error("Unexpected data format. Please check your data.")
            raise ValueError

    if args.head and args.tail:
        print_error("Head and tail cannot be set at the same time.")
        return
    nni_config = Config(get_config_filename(args))
    rest_port = nni_config.get_config("restServerPort")
    rest_pid = nni_config.get_config("restServerPid")
    if not detect_process(rest_pid):
        print_error("Experiment is not running...")
        return
    running, response = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(trial_jobs_url(rest_port), REST_TIME_OUT)
        if response and check_response(response):
            content = json.loads(response.text)
            if args.head:
                assert (
                    args.head > 0
                ), "The number of requested data must be greater than 0."
                content = sorted(
                    filter(lambda x: "finalMetricData" in x, content),
                    key=cmp_to_key(final_metric_data_cmp),
                    reverse=True,
                )[: args.head]
            elif args.tail:
                assert (
                    args.tail > 0
                ), "The number of requested data must be greater than 0."
                content = sorted(
                    filter(lambda x: "finalMetricData" in x, content),
                    key=cmp_to_key(final_metric_data_cmp),
                )[: args.tail]
            for index, value in enumerate(content):
                content[index] = convert_time_stamp_to_date(value)
            return content
        else:
            print_error("List trial failed...")
    else:
        print_error("Restful server is not running...")
    return None


def list_experiment(args):
    """Get experiment information"""
    nni_config = Config(get_config_filename(args))
    rest_port = nni_config.get_config("restServerPort")
    rest_pid = nni_config.get_config("restServerPid")
    if not detect_process(rest_pid):
        print_error("Experiment is not running...")
        return
    running, _ = check_rest_server_quick(rest_port)
    if running:
        response = rest_get(experiment_url(rest_port), REST_TIME_OUT)
        if response and check_response(response):
            content = convert_time_stamp_to_date(json.loads(response.text))
            return content
        else:
            print_error("List experiment failed...")
    else:
        print_error("Restful server is not running...")
    return None


DEFAULT_SEARCH_SPACE_PATH = os.path.split(__file__)[0] + "/" + "search_space.json"


def make_nni_experiment_config(
    experiment_name,
    search_space_path=DEFAULT_SEARCH_SPACE_PATH,
    hours=1,
    max_trials=1000,
):
    return {
        "authorName": "default",
        "experimentName": experiment_name,
        "maxExecDuration": f"{hours * 60 * 60}",  # s
        "maxTrialNum": max_trials,
        "searchSpacePath": search_space_path,
        "trainingServicePlatform": "local",
        "trial": {
            "codeDir": "/Users/maksim/dev_projects/blindml/",
            "command": "python3 -m blindml.backend.run",
            "gpuNum": 0,
        },
        "trialConcurrency": 1,
        "tuner": {
            "builtinTunerName": "TPE",
            "classArgs": {"optimize_mode": "minimize"},
        },
        "useAnnotation": False,
    }


def run_nni(experiment_config):
    experiment_name = experiment_config["experimentName"]
    stop_experiment(NniArgs(id=experiment_name, all=True))
    time.sleep(5)
    create_experiment(
        NniArgs(foreground=False, debug=False, port=8080), experiment_config
    )
    # parse_args()


def get_experiment_update(experiment_config):
    experiment_name = experiment_config["experimentName"]
    list_experiment(NniArgs(id=experiment_name))
    trials = trial_ls(NniArgs(id=experiment_name, head=False, tail=False))
    good_trials = [t for t in trials if "finalMetricData" in t]
    sorted_good_trias = sorted(
        good_trials,
        key=lambda t: float(t["finalMetricData"][0]["data"].replace('"', ""))
        if experiment_config["tuner"]["classArgs"]["optimize_mode"] == "minimize"
        else -float(t["finalMetricData"][0]["data"].replace('"', "")),
    )
    top_trial = sorted_good_trias[0]
    top_trial['hyperParameters'] = json.loads(top_trial['hyperParameters'][0])['parameters']
    return top_trial
