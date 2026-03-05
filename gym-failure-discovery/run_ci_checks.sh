#!/bin/bash
set -e
./run_autoformat.sh
mypy src/
pytest . --pylint -m pylint
pytest tests/
