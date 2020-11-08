#!/usr/bin/env bash
PYTHON=python

echo 'C3PO:'
${PYTHON} C3PO/main.py --expname "test_test" --data "C3PO/data_50/" --load_checkpoint "C3PO/checkpoints/model_50.pt" --inference "true"