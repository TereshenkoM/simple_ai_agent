set -eo pipefail

if [[ "${ENTRYPOINT_PRINTENV}" == "1" ]]; then
	    printenv | sort
fi

source /app/.venv/bin/activate
alembic upgrade head
exec python /app/main.py
