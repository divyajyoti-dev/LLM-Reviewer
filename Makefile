PYTHONPATH := $(shell pwd)/src
INPUT_JSONL ?= data/sample/examples.jsonl
OUTPUT_JSONL ?= outputs/results.jsonl

export:
	PYTHONPATH=$(PYTHONPATH) python -m reviewer_sim.ingest.gen_review_sqlite

run:
	PYTHONPATH=$(PYTHONPATH) INPUT_JSONL=$(INPUT_JSONL) OUTPUT_JSONL=$(OUTPUT_JSONL) python -m reviewer_sim.run

run-mock:
	PYTHONPATH=$(PYTHONPATH) \
	MODEL_PROVIDER=mock \
	INPUT_JSONL=data/processed/gen_review_sample.jsonl \
	OUTPUT_JSONL=outputs/results_mock.jsonl \
	python -m reviewer_sim.run

run-llamacpp:
	@if [ -z "$(MODEL_PATH)" ]; then echo "ERROR: MODEL_PATH env var is required"; exit 1; fi
	PYTHONPATH=$(PYTHONPATH) \
	MODEL_PROVIDER=llamacpp \
	MODEL_PATH=$(MODEL_PATH) \
	INPUT_JSONL=data/processed/gen_review_sample.jsonl \
	OUTPUT_JSONL=outputs/results_llamacpp.jsonl \
	python -m reviewer_sim.run

smoke:
	@head -n 1 data/processed/gen_review_sample.jsonl > data/processed/one.jsonl
	PYTHONPATH=$(PYTHONPATH) \
	MODEL_PROVIDER=mock \
	INPUT_JSONL=data/processed/one.jsonl \
	OUTPUT_JSONL=outputs/one.jsonl \
	python -m reviewer_sim.run

test-imports:
	PYTHONPATH=$(PYTHONPATH) python -c "import reviewer_sim; import reviewer_sim.ingest; import reviewer_sim.generate; import reviewer_sim.evaluate; import reviewer_sim.utils"
