set -e
#nnictl stop
#nnictl create --config config.yml
python -m blindml.runner -f /Users/maksim/dev_projects/blindml/tests/task.jsonnet

# nnictl trial ls | jq -c '.[] | select(.status | contains("FAILED"))| .stderrPath'
