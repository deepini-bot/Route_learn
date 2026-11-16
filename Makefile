.PHONY: test demo experiments serve
test:        ; python -m unittest discover -s tests -v
demo:        ; python examples/demo.py
experiments: ; python experiments/run_experiments.py
serve:       ; uvicorn routelearn.server:app --reload
