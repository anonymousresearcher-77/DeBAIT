#!/usr/bin/env bash
set -euo pipefail


EVAL_DATASETS="${EVAL_DATASETS:-FairMTBench BBQ ImplicitBBQ BOLD ToxiGen}"
BBQ_EVAL_CATEGORIES="${BBQ_EVAL_CATEGORIES:-Gender_identity Race_ethnicity Race_x_SES Religion SES Sexual_orientation Physical_appearance Nationality}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-final_paper_results/loss_sweep}"
RESULTS_CSV="${RESULTS_CSV:-${OUTPUT_ROOT}/results.csv}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_ROOT}/.cache}"

ITER_REASON="150"
ITER_ANS="10"

BOLD_SAMPLES="${BOLD_SAMPLES:-10000}"
TOXIGEN_SAMPLES="${TOXIGEN_SAMPLES:-5000}"
FAIRMTBENCH_SAMPLES="${FAIRMTBENCH_SAMPLES:-500}"

IPO_TAU_DEFAULT="${IPO_TAU_DEFAULT:-0.1}"
IPO_TAU_SENSITIVE="${IPO_TAU_SENSITIVE:-0.05}"
MODELS="${MODELS:-}"

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

BIASDPO_ARGS="--train-dataset BiasDPO \
    --biasdpo-dir data/bias-grpo-data/bias-dpo"

ALLBIAS_ARGS="--train-dataset AllBias \
    --biasdpo-dir data/bias-grpo-data/bias-dpo \
    --civil-comments-path data/bias-grpo-data/civil-comments/train-00000-of-00001.parquet \
    --civil-toxicity-threshold 0.5"

COMMON_ARGS="
    --eval-datasets       ${EVAL_DATASETS}
    --bbq-eval-categories ${BBQ_EVAL_CATEGORIES}
    --epochs              ${EPOCHS}
    --lr                  ${LR}
    --batch-size          ${BATCH_SIZE}
    --results-csv         ${RESULTS_CSV}
    --use-instruction
    --cache-dir           ${CACHE_DIR}
    --bold-n-samples                 ${BOLD_SAMPLES}
    --run-contamination-check \
    --contamination-calibration-path data/calibration/post_cutoff_articles.jsonl \
    --contamination-min-k-percent 20 \
    --contamination-max-items 300 \
    --contamination-ts-guessing-samples 200 \
"


run_exp() {
    local run_id="$1"; shift

    echo ""
    echo "============================================================"
    echo "  RUNNING: ${run_id}"
    echo "============================================================"

    if python main.py ${COMMON_ARGS} \
        --output-dir "${OUTPUT_ROOT}/${run_id}" \
        --run-id "${run_id}" \
        "$@" \
        2>&1 | tee "${LOG_DIR}/${run_id}.log"; then

        echo "✓ ${run_id} completed."

    else
        echo "✗ ${run_id} failed. Continuing..."
    fi
}
for MODEL in ${MODELS}; do

    M="${MODEL//\//-}"
    M="${M//_/-}"

    echo ""
    echo "###################################################################"
    echo "  MODEL: ${MODEL}  (slug: ${M})"
    echo "###################################################################"


    run_exp "${M}__ipo__baseline__AllBias" \
        --model-id "${MODEL}" \
        --loss-fn ce_ipo \
        --ipo-tau "${IPO_TAU_DEFAULT}" \
        ${ALLBIAS_ARGS}

    run_exp "${M}__ipo__guardrail-ti__role-ep__AllBias" \
        --model-id "${MODEL}" \
        --loss-fn ce_ipo \
        --ipo-tau "${IPO_TAU_DEFAULT}" \
        --train-instructions guardrail \
        --eval-prompts role \
        ${ALLBIAS_ARGS}

    run_exp "${M}__ipo__intervention-ti__role-ep__AllBias" \
        --model-id "${MODEL}" \
        --loss-fn ce_ipo \
        --ipo-tau "${IPO_TAU_DEFAULT}" \
        --train-instructions intervention \
        --eval-prompts role \
        ${ALLBIAS_ARGS}

    run_exp "${M}__ipo__intervention-ti__qif-ep__AllBias" \
        --model-id "${MODEL}" \
        --loss-fn ce_ipo \
        --ipo-tau "${IPO_TAU_DEFAULT}" \
        --train-instructions intervention \
        --eval-prompts q_if \
        ${ALLBIAS_ARGS}

    run_exp "${M}__ipo__causal-ti__role-ep__AllBias" \
        --model-id "${MODEL}" \
        --loss-fn ce_ipo \
        --ipo-tau "${IPO_TAU_DEFAULT}" \
        --train-instructions causal_instr \
        --eval-prompts role \
        ${ALLBIAS_ARGS}
        
    run_exp "${M}__ipo__guardrail-ti__guardrail-ep__AllBias" \
        --model-id "${MODEL}" \
        --loss-fn ce_ipo \
        --ipo-tau "${IPO_TAU_DEFAULT}" \
        --train-instructions guardrail \
        --eval-prompts guardrail_eval \
        ${ALLBIAS_ARGS}
done

echo ""
echo "###################################################################"
echo "  ALL RUNS COMPLETE"
echo "  Results CSV : ${RESULTS_CSV}"
echo "  Logs        : ${LOG_DIR}/"
echo "  Cache       : ${CACHE_DIR}/"
echo "###################################################################"