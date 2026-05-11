/bin/ollama serve &
pid=$!

sleep 5

MODEL_TO_PULL="${LLM_MODEL:-llama3.1:8b}"

if ! ollama list | awk '{print $1}' | grep -qx "${MODEL_TO_PULL}"; then
  ollama pull "${MODEL_TO_PULL}"
fi

wait $pid
