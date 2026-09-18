#!/bin/zsh
# T10 mutation gate driver: backup-first, kill, byte-identical restore.
# Evidence lands under ~/Workspace/solweig_ultrafast_artifacts/t10/mutations/.
set -u
WT=/Users/alansynn/Workspace/solweig/.claude/worktrees/agent-aefc95765220c3002
cd "$WT"
PY=/Users/alansynn/Workspace/solweig/.venv/bin/python
T=$WT/solweig_core/numba_cpu/thermal.py
MUT=/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t10/mutations
PRISTINE=$MUT/thermal.py.pristine

# id | designated kill test (node id in tests/ultrafast/test_thermal.py)
typeset -a IDS KILLS
IDS=(M1 M2 M3 M4 M5 M6)
KILLS=(
  "TestR1GeometryHistorySeparation::test_previous_scene_noon_checkpoint_refused_after_geometry_edit"
  "TestR4MetPrefixInvalidation::test_digest_spelled_at_candidate_own_next_step"
  "TestResumeAndScatter::test_replay_resumes_exactly_at_next_step"
  "TestR2CarriedScalars::test_twater_absent_carried_as_not_established_never_defaulted"
  "TestResumeAndScatter::test_selected_time_outputs_scatter_to_global_t"
  "TestR6TornState::test_digest_fence_rejects_bit_flips"
)

overall=0
for i in $(seq 1 ${#IDS[@]}); do
  id=${IDS[$i]}
  kill=${KILLS[$i]}
  log=$MUT/${id}_run.txt
  {
    echo "==== mutation $id — designated kill: $kill ===="
    # 1. live file is pristine before we start
    shasum -a 256 "$T"
    if ! cmp -s "$T" "$PRISTINE"; then
      echo "FATAL: live thermal.py is not the pristine backup"; exit 9
    fi
    # 2. apply the mutation
    $PY "$WT/tests/ultrafast/mutate_thermal.py" "$id" || exit 9
    # 3. designated test must FAIL
    $PY -m pytest "tests/ultrafast/test_thermal.py::$kill" -q --no-header -x >/dev/null 2>&1
    kill_rc=$?
    echo "kill_rc=$kill_rc (0 means SURVIVED — gate failure)"
    [ $kill_rc -ne 0 ] || { overall=1; echo "SURVIVED: $id"; }
    # 4. restore byte-identical, sha-verified
    cp "$PRISTINE" "$T"
    shasum -a 256 "$T"
    cmp -s "$T" "$PRISTINE" || { echo "FATAL: restore not byte-identical"; exit 9; }
    # 5. designated test must PASS again
    $PY -m pytest "tests/ultrafast/test_thermal.py::$kill" -q --no-header >/dev/null 2>&1
    pass_rc=$?
    echo "restore_pass_rc=$pass_rc (0 required)"
    [ $pass_rc -eq 0 ] || { overall=1; echo "RESTORE-BROKEN: $id"; }
    echo "==== $id done ===="
  } >> "$log" 2>&1
  echo "---- $id -> $log"
  tail -n 4 "$log"
done
echo "overall=$overall (0 = all mutations killed, restores verified)"
exit $overall
