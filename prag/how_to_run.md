1.vllm
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.8

2.ollama
ollama serve

3.Collaborative_Edge
python Collaborative_Edge.py

4.test
python test_privacy.py