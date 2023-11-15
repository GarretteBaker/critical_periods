#!/bin/bash

python experiment_works.py > output.txt 2>&1
runpodctl stop pod apds5u8rwntfzv