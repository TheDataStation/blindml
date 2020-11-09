from pprint import pprint

from gooey import GooeyParser

from blindml.frontend.config.task.task import parse_task_capsule


# this function should either parse a jsonnet or create one using a gui.
# nothing else. leave the running of stuff to the task class itself
# @Gooey(target="ffmpeg", program_name="Frame Extraction v1.0", suppress_gooey_flag=True)
def main():
    parser = GooeyParser(description="BlindML")
    parser.add_argument(
        "-f",
        default="/Users/maksim/dev_projects/blindml/tests/task.jsonnet",
        help="Task capsule file path",
        dest="task_file_fp",
    )
    args = parser.parse_args()

    task = parse_task_capsule(args.task_file_fp)
    # task.run()
    # pprint(task.get_experiment_update())
    task.save_best_model("/Users/maksim/dev_projects/blindml/tests")

if __name__ == "__main__":
    main()
