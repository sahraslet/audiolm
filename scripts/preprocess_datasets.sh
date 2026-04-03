#!/bin/bash
# run_all_preprocessing.sh
# runs all preprocessing scripts
# even if it crashes, does not rerun everything only the failed one

run_if_not_done() {
    local script=$1
    local done_flag=$2

    if [ -f "$done_flag" ]; then
        echo "Skipping $script — already done."
    else
        echo "Running $script..."
        bash $script
        if [ $? -ne 0 ]; then
            echo "ERROR: $script failed!"
            exit 1
        fi
    fi
}

echo "=== Starting preprocessing pipeline ==="

run_if_not_done scripts/generate_asr_english.sh      data/asr_english/.done
run_if_not_done scripts/generate_asr_german.sh       data/asr_german/.done
run_if_not_done scripts/run_s2st_preprocessing.sh    data/s2st/.done
run_if_not_done scripts/generate_t2t_dataset.sh      data/t2t/.done

echo "=== Merging datasets ==="
if [ -f "data/all/.done" ]; then
    echo "Skipping merge — already done."
else
    python src/audiolm/merge_datasets.py \
        --input_dirs data/asr_english data/asr_german data/s2st data/t2t \
        --output_dir data/all \
        --seed 1337 \
        --push_to_hub \
        --hub_repo_id sahara22/audiolm-eval
fi

echo "=== Preprocessing pipeline complete ==="
echo "Dataset ready at data/all"