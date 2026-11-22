#!/usr/bin/env bash
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
# GPU lane: holds one H200 and runs the campaigns staged for it one after the other, so the GPU
# never goes back to the queue between campaigns or while a fix is prepared.
#
#   sbatch --job-name=gpu-lane-NAME --output=LANE/slurm-%j.out --export=ALL,LANE=LANE \
#          [--dependency=afterany:JOB] lane.sbatch
#
# LANE/stages has one stage per line, "NAME COMMAND..." (blank lines and # comments ignored), and
# is re-read before every stage, so stages can be added while the lane runs. COMMAND is exec'd
# (one script or program with its arguments), so the lane's signals reach the stage itself.
# LANE/state/NAME.done records a finished stage's exit status: 0 done, 1 done with failed cases,
# anything else after two failed runs (counted in LANE/state/NAME.failures). A stage stopped by
# the time-limit warning (it saves and exits 2) or by scancel is resumed by the next lane job. Stages must resume from their own records and never requeue
# the job themselves (a requeued job is ineligible for two minutes and loses the GPU), so the
# stage scripts' restart counters are forced to 0.
#
# With no stage left the lane waits, polling every minute, for LANE/idle_minutes (default 360,
# re-read while waiting) and then releases the GPU. At the time-limit warning the lane first
# submits its successor (afterany on this job: it takes the GPU the moment this one ends), then
# asks the stage to save and stop. scancel (SIGTERM) stops the stage and cancels the successor.
set -uo pipefail
: "${LANE:?set LANE to the lane directory}"
# Submit from the repository root, or set LANE_SCRIPT.
LANE_SCRIPT=${LANE_SCRIPT:-${SLURM_SUBMIT_DIR:-$PWD}/campaigns/gpu_lane/lane.sbatch}
export HYBRIDGE_GC_MAX_RESTARTS=0 HYBRIDGE_UNIFIED_MAX_RESTARTS=0 HYBRIDGE_Q1_MAX_RESTARTS=0
mkdir -p "$LANE/state"
log() { echo "[$(date -Is)] lane $(basename "$LANE") job ${SLURM_JOB_ID:-none}: $*"; }
log "started on $(hostname), restart ${SLURM_RESTART_COUNT:-0}, time limit $(squeue -h -j "${SLURM_JOB_ID:-0}" -o %l 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>&1 || true

stage_pid='' successor_id='' warned='' stopping=''

submit_successor() {
    [[ -n $successor_id ]] && return
    local clean=() var
    for var in $(compgen -e); do
        [[ $var == SLURM_* || $var == SBATCH_* || $var == CUDA_VISIBLE_DEVICES || $var == GPU_DEVICE_ORDINAL ]] && clean+=(-u "$var")
    done
    # shellcheck disable=SC2086  # LANE_SBATCH_OPTS holds extra sbatch options, word-split on purpose
    if successor_id=$(env "${clean[@]}" sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
            --job-name="$SLURM_JOB_NAME" --output="$LANE/slurm-%j.out" --export=ALL,LANE="$LANE" \
            ${LANE_SBATCH_OPTS:-} "$LANE_SCRIPT" 2>&1); then
        successor_id=${successor_id%%;*}
        log "successor $successor_id submitted (afterany:$SLURM_JOB_ID)"
    else
        log "successor submission FAILED: $successor_id"
        successor_id=''
    fi
}

on_warning() {
    warned=1
    log 'time-limit warning'
    # Hand the GPU on only when there is work left; an idle lane just ends.
    if [[ -n $stage_pid ]] || next_stage > /dev/null; then submit_successor; fi
    [[ -n $stage_pid ]] && { log "asking stage (pid $stage_pid) to save and stop"; kill -USR1 "$stage_pid" 2>/dev/null; }
}

on_stop() {
    stopping=1
    # After the warning this is Slurm's time-limit kill: the successor must still take over.
    if [[ -z $warned ]]; then
        log 'stop requested'
        if [[ -n $successor_id ]]; then scancel "$successor_id" && log "successor $successor_id cancelled"; fi
    fi
    [[ -n $stage_pid ]] && kill -TERM -- "-$stage_pid" 2>/dev/null
}
trap on_warning USR1
trap on_stop TERM INT HUP

next_stage() {  # prints "NAME<TAB>COMMAND" for the first stage without a done marker
    [[ -f $LANE/stages ]] || return 1
    local name command
    while read -r name command; do
        [[ -z $name || $name == \#* ]] && continue
        [[ -f $LANE/state/$name.done ]] && continue
        printf '%s\t%s\n' "$name" "$command"
        return 0
    done < "$LANE/stages"
    return 1
}

idle_since=$(date +%s) reported=0
while [[ -z $warned && -z $stopping ]]; do
    if line=$(next_stage); then
        name=${line%%$'\t'*} command=${line#*$'\t'}
        failures=$(cat "$LANE/state/$name.failures" 2>/dev/null || echo 0)
        log "stage $name (failed runs so far: $failures): $command"
        # Own session: the stage and everything it starts form one process group (pgid = pid).
        LANE_STAGE=$name setsid bash -c "exec $command" &
        stage_pid=$!
        wait "$stage_pid"; status=$?
        # A trapped signal interrupts wait; keep waiting for the stage's own exit.
        while kill -0 "$stage_pid" 2>/dev/null; do wait "$stage_pid"; status=$?; done
        # Sweep anything the stage left behind in its group before the next stage gets the GPU.
        if kill -0 -- "-$stage_pid" 2>/dev/null; then
            log "stage $name left processes; terminating its group"
            kill -TERM -- "-$stage_pid" 2>/dev/null; sleep 10; kill -KILL -- "-$stage_pid" 2>/dev/null
        fi
        stage_pid=''
        log "stage $name exited $status"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  GPU process: /'
        if [[ $status == 0 || $status == 1 ]]; then
            echo "$status" > "$LANE/state/$name.done"
        elif [[ -n $warned || -n $stopping ]]; then
            log "stage $name stopped (exit $status); the next lane job resumes it"
        else
            failures=$(( failures + 1 ))
            echo "$failures" > "$LANE/state/$name.failures"
            if (( failures >= 2 )); then
                echo "$status" > "$LANE/state/$name.done"
                log "stage $name failed twice (exit $status); moving on"
            else
                log "stage $name failed (exit $status); retrying once"
            fi
        fi
        idle_since=$(date +%s) reported=0
        continue
    fi
    limit=$(cat "$LANE/idle_minutes" 2>/dev/null || echo 360)
    [[ $limit =~ ^[0-9]+$ ]] || limit=360
    idle=$(( ($(date +%s) - idle_since) / 60 ))
    if (( idle >= limit )); then
        log "no stage for $idle min (limit $limit); releasing the GPU"
        break
    fi
    if (( idle >= reported )); then
        log "waiting for a stage in $LANE/stages (idle $idle min, limit $limit)"
        reported=$(( idle + 30 ))
    fi
    sleep 60 & wait $!
done
log "exiting (warned=${warned:-no}, stopped=${stopping:-no}, successor=${successor_id:-none})"
exit 0
