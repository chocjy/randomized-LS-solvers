#!/bin/bash

case "$1" in
    --help|-h) spark-submit --driver-java-options '-Dlog4j.configuration=log4j.properties' run_ls.py print_help
        ;;
    *) spark-submit --driver-java-options '-Dlog4j.configuration=log4j.properties' --executor-memory 5G --driver-memory 5G --py-files comp_sketch.py,ls_utils.py,utils.py,rowmatrix.py,projections.py,sampling.py run_ls.py $@ 
esac
