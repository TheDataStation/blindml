import json
import os
import random
import signal
import string
import sys
import time
from functools import cmp_to_key
from pprint import pprint
from subprocess import check_call, CalledProcessError, Popen, PIPE, STDOUT, call

import psutil
from nni.package_utils import get_builtin_module_class_name, get_nni_installation_path
from nni_cmd.common_utils import print_error, print_normal, detect_process, detect_port
from nni_cmd.config_utils import Config, Experiments
from nni_cmd.constants import (
    INSTALLABLE_PACKAGE_META,
    ERROR_INFO,
    REST_TIME_OUT,
    LOG_HEADER,
)
from nni_cmd.launcher import (
    get_log_path,
    print_log_content,
    set_platform_config,
    set_experiment,
)
from nni_cmd.nnictl_utils import (
    get_config_filename,
    convert_time_stamp_to_date,
    # stop_experiment,
    update_experiment,
    parse_ids,
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
        # TODO: this method mutates nni_config
        # TODO: i had to hack nni/main.js
        launch_experiment(
            args, experiment_config, "new", experiment_name, experiment_name
        )
    except Exception as e:
        nni_config = Config(experiment_name)
        rest_server_pid = nni_config.get_config("restServerPid")
        if rest_server_pid:
            kill_command(rest_server_pid)
        raise e


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


def make_nni_experiment_config(experiment_name, search_space, hours=1, max_trials=1000):
    here = os.path.dirname(os.path.abspath(__file__))
    return {
        "authorName": "default",
        "experimentName": experiment_name,
        "maxExecDuration": f"{hours * 60 * 60}",  # s
        "maxTrialNum": max_trials,
        "trainingServicePlatform": "local",
        "trial": {
            "codeDir": f"{here}/../../../blindml/",
            "command": "source venv/bin/activate && python3 -m blindml.backend.run",
            # "gpuNum": 0,
        },
        "trialConcurrency": 1,
        "tuner": {
            "builtinTunerName": "TPE",
            "classArgs": {"optimize_mode": "minimize"},
        },
        "useAnnotation": False,
        # nni expects search space to be a string (stupid)
        "searchSpace": json.dumps(search_space),
    }


def run_nni(experiment_config):
    experiment_name = experiment_config["experimentName"]
    stop_experiment(NniArgs(id=experiment_name, all=True))
    time.sleep(5)
    create_experiment(
        NniArgs(foreground=False, debug=False, port=8080), experiment_config
    )


def resume(experiment_config):
    experiment_name = experiment_config["experimentName"]
    resume_experiment(
        NniArgs(
            id=experiment_name,
            head=False,
            tail=False,
            foreground=False,
            debug=False,
            port=8080,
        )
    )


def get_experiment_update(experiment_config):
    experiment_name = experiment_config["experimentName"]
    list_experiment(NniArgs(id=experiment_name))
    trials = trial_ls(NniArgs(id=experiment_name, head=False, tail=False))
    good_trials = [t for t in trials if t["status"] == "SUCCEEDED"]
    for t in good_trials:
        t["finalMetricData"] = json.loads(json.loads(t["finalMetricData"][0]["data"]))
    sorted_good_trias = sorted(
        good_trials,
        key=lambda t: t["finalMetricData"]["default"]
        if experiment_config["tuner"]["classArgs"]["optimize_mode"] == "minimize"
        else -t["finalMetricData"]["default"],
    )
    for trial in sorted_good_trias:
        trial["hyperParameters"] = json.loads(trial["hyperParameters"][0])["parameters"]
    return sorted_good_trias


def manage_stopped_experiment(args, mode):
    update_experiment()
    experiment_config = Experiments()
    experiment_dict = experiment_config.get_all_experiments()
    experiment_id = None
    # find the latest stopped experiment
    if not args.id:
        print_error(
            "Please set experiment id! \nYou could use 'nnictl {0} id' to {0} a stopped experiment!\n"
            "You could use 'nnictl experiment list --all' to show all experiments!".format(
                mode
            )
        )
        exit(1)
    else:
        if experiment_dict.get(args.id) is None:
            print_error("Id %s not exist!" % args.id)
            exit(1)
        if experiment_dict[args.id]["status"] != "STOPPED":
            print_error("Only stopped experiments can be {0}ed!".format(mode))
            exit(1)
        experiment_id = args.id
    print_normal("{0} experiment {1}...".format(mode, experiment_id))
    nni_config = Config(experiment_dict[experiment_id]["fileName"])
    experiment_config = nni_config.get_config("experimentConfig")
    experiment_id = nni_config.get_config("experimentId")
    experiment_name = experiment_config["experimentName"]
    new_config_file_name = "".join(
        random.sample(string.ascii_letters + string.digits, 8)
    )
    new_nni_config = Config(new_config_file_name)
    new_nni_config.set_config("experimentConfig", experiment_config)
    new_nni_config.set_config("restServerPort", args.port)
    try:
        launch_experiment(
            args, experiment_config, mode, experiment_name, experiment_name
        )
    except Exception as exception:
        nni_config = Config(new_config_file_name)
        restServerPid = nni_config.get_config("restServerPid")
        if restServerPid:
            kill_command(restServerPid)
        print_error(exception)
        exit(1)


def view_experiment(args):
    """view a stopped experiment"""
    manage_stopped_experiment(args, "view")


def resume_experiment(args):
    """resume an experiment"""
    manage_stopped_experiment(args, "resume")


def get_experiment_id(experiment_name):
    return list_experiment(
        NniArgs(id=experiment_name, foreground=False, debug=False, port=8080)
    )["id"]


def start_rest_server(
    port,
    platform,
    mode,
    config_file_name,
    foreground=False,
    experiment_id=None,
    log_dir=None,
    log_level=None,
):
    """Run nni manager process"""
    if detect_port(port):
        print_error(
            "Port %s is used by another process, please reset the port!\n"
            "You could use 'nnictl create --help' to get help information" % port
        )
        exit(1)

    if (platform != "local") and detect_port(int(port) + 1):
        print_error(
            "PAI mode need an additional adjacent port %d, and the port %d is used by another process!\n"
            "You could set another port to start experiment!\n"
            "You could use 'nnictl create --help' to get help information"
            % ((int(port) + 1), (int(port) + 1))
        )
        exit(1)

    print_normal("Starting restful server...")

    entry_dir = get_nni_installation_path()
    if (not entry_dir) or (not os.path.exists(entry_dir)):
        print_error("Fail to find nni under python library")
        exit(1)
    entry_file = os.path.join(entry_dir, "main.js")
    node_command = "node"
    if sys.platform == "win32":
        node_command = os.path.join(entry_dir[:-3], "Scripts", "node.exe")
    cmds = [
        node_command,
        "--max-old-space-size=4096",
        entry_file,
        "--port",
        str(port),
        "--mode",
        platform,
    ]
    if mode == "view":
        cmds += ["--start_mode", "resume"]
        cmds += ["--readonly", "true"]
    else:
        cmds += ["--start_mode", mode]
    if log_dir is not None:
        cmds += ["--log_dir", log_dir]
    if log_level is not None:
        cmds += ["--log_level", log_level]
    cmds += ["--experiment_id", experiment_id]
    if foreground:
        cmds += ["--foreground", "true"]
    stdout_full_path, stderr_full_path = get_log_path(config_file_name)
    with open(stdout_full_path, "a+") as stdout_file, open(
        stderr_full_path, "a+"
    ) as stderr_file:
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        # add time information in the header of log files
        log_header = LOG_HEADER % str(time_now)
        stdout_file.write(log_header)
        stderr_file.write(log_header)
        if sys.platform == "win32":
            from subprocess import CREATE_NEW_PROCESS_GROUP

            if foreground:
                process = Popen(
                    cmds,
                    cwd=entry_dir,
                    stdout=PIPE,
                    stderr=STDOUT,
                    creationflags=CREATE_NEW_PROCESS_GROUP,
                )
            else:
                process = Popen(
                    cmds,
                    cwd=entry_dir,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    creationflags=CREATE_NEW_PROCESS_GROUP,
                )
        else:
            if foreground:
                process = Popen(cmds, cwd=entry_dir, stdout=PIPE, stderr=PIPE)
            else:
                process = Popen(
                    cmds, cwd=entry_dir, stdout=stdout_file, stderr=stderr_file
                )
    return process, str(time_now)


def stop_experiment(args):
    """Stop the experiment which is running"""
    experiment_id_list = parse_ids(args)
    if experiment_id_list:
        experiment_config = Experiments()
        experiment_dict = experiment_config.get_all_experiments()
        for experiment_id in experiment_id_list:
            print_normal("Stopping experiment %s" % experiment_id)
            nni_config = Config(experiment_dict[experiment_id]["fileName"])
            rest_pid = nni_config.get_config("restServerPid")
            if rest_pid:
                kill_command(rest_pid)
                tensorboard_pid_list = nni_config.get_config("tensorboardPidList")
                if tensorboard_pid_list:
                    for tensorboard_pid in tensorboard_pid_list:
                        try:
                            kill_command(tensorboard_pid)
                        except Exception as exception:
                            print_error(exception)
                    nni_config.set_config("tensorboardPidList", [])
            print_normal("Stop experiment success.")
            experiment_config.update_experiment(experiment_id, "status", "STOPPED")
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            experiment_config.update_experiment(experiment_id, "endTime", str(time_now))


def kill_command(pid):
    """kill command"""
    if sys.platform == "win32":
        process = psutil.Process(pid=pid)
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        cmds = ["kill", "-9", str(pid)]
        call(cmds)


if __name__ == "__main__":
    pprint(make_nni_experiment_config("test", {}))
