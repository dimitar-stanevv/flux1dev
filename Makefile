IMAGE ?= ghcr.io/dimitar-stanevv/flux1dev
TAG   ?= latest

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[1;34m%-16s\033[0m %s\n", $$1, $$2}'

build: ## Build the RunPod image (CI does this for you — see .github/workflows)
	docker build --platform linux/amd64 -t $(IMAGE):$(TAG) .

push: build ## Build and push to GHCR
	docker push $(IMAGE):$(TAG)

run: ## Run locally (needs an NVIDIA GPU + nvidia-container-toolkit)
	docker run --rm -it --gpus all \
		-p 8188:8188 \
		-v $(PWD)/.volume:/workspace \
		-e HF_TOKEN=$${HF_TOKEN:-} \
		-e LORAS=$${LORAS:-} \
		$(IMAGE):$(TAG)

shell: ## Shell into the image without starting ComfyUI
	docker run --rm -it --gpus all -v $(PWD)/.volume:/workspace \
		--entrypoint bash $(IMAGE):$(TAG)

workflows: ## Regenerate the two workflow graphs (validates links before writing)
	python3 tools/gen_workflows.py

check: ## Lint the shell scripts and validate the Python + workflow JSON
	@command -v shellcheck >/dev/null && shellcheck -S warning scripts/*.sh || echo "shellcheck not installed, skipping"
	@bash -n scripts/entrypoint.sh && bash -n scripts/bootstrap.sh && echo "shell: syntax ok"
	@python3 -m py_compile scripts/provision.py tools/gen_workflows.py && echo "python: compiles"
	@python3 -c "import json,glob; [json.load(open(p)) for p in glob.glob('workflows/*.json')]; print('workflows: valid JSON')"
	@rm -rf scripts/__pycache__ tools/__pycache__

.PHONY: help build push run shell workflows check
