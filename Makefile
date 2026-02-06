PYTHONPATH := $(shell pwd)/src
INPUT_JSONL ?= data/processed/review_subset.jsonl
OUTPUT_JSONL ?= outputs/results.jsonl

# Export clean subset from SQLite
export:
	PYTHONPATH=$(PYTHONPATH) python -m reviewer_sim.ingest.export_review_subset \
		--db-path data/gen_review.db \
		--out-path data/processed/review_subset.jsonl \
		--n 200 --seed 42 --min-year 2021 --min-review-chars 50

# Enrich with LLM-classified primary areas (requires TOGETHER_API_KEY)
enrich:
	PYTHONPATH=$(PYTHONPATH) python -m reviewer_sim.ingest.enrich_primary_area \
		--in-path data/processed/review_subset.jsonl \
		--out-path data/processed/review_subset_enriched.jsonl \
		--model "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Run review simulation pipeline
run:
	PYTHONPATH=$(PYTHONPATH) INPUT_JSONL=$(INPUT_JSONL) OUTPUT_JSONL=$(OUTPUT_JSONL) python -m reviewer_sim.run

# Run with mock generator
run-mock:
	PYTHONPATH=$(PYTHONPATH) \
	MODEL_PROVIDER=mock \
	INPUT_JSONL=data/processed/review_subset.jsonl \
	OUTPUT_JSONL=outputs/results_mock.jsonl \
	python -m reviewer_sim.run

# Run with llamacpp (requires MODEL_PATH)
run-llamacpp:
	@if [ -z "$(MODEL_PATH)" ]; then echo "ERROR: MODEL_PATH env var is required"; exit 1; fi
	PYTHONPATH=$(PYTHONPATH) \
	MODEL_PROVIDER=llamacpp \
	MODEL_PATH=$(MODEL_PATH) \
	INPUT_JSONL=data/processed/review_subset.jsonl \
	OUTPUT_JSONL=outputs/results_llamacpp.jsonl \
	python -m reviewer_sim.run

# Run tests
test:
	PYTHONPATH=$(PYTHONPATH) pytest tests/ -v

# Check imports
test-imports:
	PYTHONPATH=$(PYTHONPATH) python -c "import reviewer_sim; import reviewer_sim.ingest; import reviewer_sim.generate; import reviewer_sim.evaluate; import reviewer_sim.utils"

# Summarize results
summarize:
	PYTHONPATH=$(PYTHONPATH) python scripts/summarize_results.py $(OUTPUT_JSONL)
