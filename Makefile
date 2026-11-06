setup:      ## install deps + pre-commit
	uv sync && uv run pre-commit install
up:         ## start OpenSearch
	docker compose up -d opensearch
check:      ## lint + types + unit tests
	uv run ruff check . && uv run ruff format --check . && uv run mypy ragantine bench && uv run pytest -m "not integration"
corpus:
	uv run ragantine corpus build
index:
	uv run ragantine index build --no-poison
bench-data:
	uv run ragantine bench questions && uv run ragantine bench poison
bench:      ## baseline on test split
	uv run ragantine bench run --defenses none --sets target,clean --split test
ablation:
	uv run ragantine bench ablation --models open-large,open-small,closed-small
report:
	uv run ragantine bench report --runs latest
serve:
	uv run ragantine serve
demo:       ## build the demo snapshot if missing, then launch Gradio
	test -f demo/snapshot/snapshot.json || uv run python demo/build_snapshot.py
	uv run python demo/app.py

.PHONY: setup up check corpus index bench-data bench ablation report serve demo job

# long benchmark jobs: detached and memory-capped (see scripts/bg.sh)
job:        ## make job NAME=questions ARGS="bench questions --config configs/v01-free.yaml --yes"
	scripts/bg.sh $(NAME) $(ARGS)
