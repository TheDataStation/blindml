set -e
nnictl stop
nnictl create --config config.yml
# nnictl trial ls | jq -c '.[] | select(.status | contains("FAILED"))| .stderrPath'
