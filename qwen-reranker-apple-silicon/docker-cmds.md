```bash
cd qwen-reranker-apple-silicon
docker buildx build --platform linux/arm64 -t hf-fastapi-m1 .
docker run --rm -it -p 8000:8000 hf-fastapi-m1
```