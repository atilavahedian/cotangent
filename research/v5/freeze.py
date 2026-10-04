"""Preregister one powered replication; the V4 algorithm remains byte-for-byte unchanged."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from cotangent.runtime import source_hash as v1_hash
from research.v2.pilot import source_hash as v2_hash
from research.v3.pilot import source_hash as v3_hash
from research.v4.pilot import source_hash as v4_hash
from research.v5.runtime import source_hash as v5_hash

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    output = ROOT / 'artifacts/v5/frozen-study.json'
    assert not output.exists(), 'Never overwrite a frozen study'
    previous = json.loads((ROOT/'artifacts/v4/frozen-study.json').read_text())
    result_path = ROOT/'artifacts/v4/analysis/results.json'
    result = json.loads(result_path.read_text())
    restores_path = ROOT/'artifacts/v4/verification-snapshots.json'
    restores = json.loads(restores_path.read_text())
    planning_path = ROOT/'artifacts/v5/power-planning.json'
    planning = json.loads(planning_path.read_text())
    assert result['runs']==70 and result['integrity_checks_passed'] and result['checkpoint_file_hashes_verified']
    assert result['equivalence_gate_pass'] and not result['primary_success']
    assert restores['status']=='passed' and restores['snapshots']==70
    assert v4_hash()==previous['source_sha256']==result['source_sha256']
    assert v3_hash()==previous['v3_source_sha256'] and v2_hash()==previous['v2_source_sha256']
    assert v1_hash(ROOT)==previous['v1_source_sha256']
    assert digest(ROOT/'data/manifest.json')==previous['data_manifest_sha256']
    assert planning['planned_pairs']==100 and planning['confidence_level']==.995
    seeds=list(range(1001,1101))
    old_seeds={j['seed'] for p in (ROOT/'artifacts').glob('v*/frozen-study.json') for j in json.loads(p.read_text())['jobs']}
    assert not set(seeds)&old_seeds
    blocks=[(seed,index%2) for index,seed in enumerate(seeds)]
    random.Random(2026100405).shuffle(blocks)
    jobs=[]
    for seed,reverse in blocks:
        arms=['native-fused','native-bucket']
        if reverse: arms.reverse()
        jobs.extend(dict(size='small',seed=seed,arm=arm,comparison='primary') for arm in arms)
    frozen=dict(previous)
    frozen.update(version=5,method_version=4,method_unchanged=True,
        frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=v5_hash(),v4_source_sha256=v4_hash(),
        configs={'small':previous['configs']['small']},jobs=jobs,primary_seeds=seeds,medium_seeds=[],
        equivalence_seeds=previous['equivalence_seeds'],confidence_level=.995,
        two_sided_student_t_critical=planning['two_sided_critical'],degrees_of_freedom=99,
        prior_result_sha256=digest(result_path),prior_restore_sha256=digest(restores_path),
        power_planning_sha256=digest(planning_path),equivalence_prerequisite=True,
        decision='All 100 fresh primary pairs must cross the target. Mean paired target elapsed saving >=10%, its two-sided 99.5% Student-t lower bound >0, and upper paired full-test difference <=0.01 BPB. The immutable V4 seven-pair full-training equivalence proof is a required prerequisite. All 200 planned runs included. No optional stopping, seed removal, adding seeds after results, threshold relaxation or secondary substitution. This is the single fixed-size powered replication of unchanged V4, not an extension or relabeling of V4.',
        inference='Two-sided 99.5% paired Student-t intervals over exactly 100 independent fresh seeds, df99 critical 2.8713076612147663. One fixed-size confirmatory replication after four disclosed failed studies; these intervals are not a universal family-wise correction for every exploratory decision. First scheduled 100-step/65,536-byte probe <=3.2 BPB defines target, no interpolation or imputation. Only the unchanged small-model/native-backprop comparison is confirmatory; no new medium or alternative-baseline results can rescue it.',
        sample_size='Full V4 quality-difference sample SD 0.020926128365766364 BPB. Normal approximation at planned zero mean difference, 0.01 margin, 99.5% two-sided interval and 90% power gives 73.2022 pairs. N=100 is frozen before any V5 outcome to exceed this estimate and allow non-normal tails. Planning does not guarantee success; no post-result extension is permitted.',
        timing=previous['timing'].replace('All 35 pair blocks shuffled, AB/BA balanced separately within comparison types','All 100 fresh pair blocks shuffled, exactly 50 AB and 50 BA'),
        heldout_scope='The method, architecture, native fused baseline, precision, budgets, hyperparameters, corpus and 10%/0.01 thresholds are unchanged from fully disclosed failed V4. Only independent seeds, fixed sample size and a stricter confidence level change. The sample size uses all V4 variance. WikiText-2 official validation/test are openly reused for numerical/quality regression, not unseen-dataset generalization. No V5 official evaluation precedes this freeze.',
        deterministic_configuration='Both V5 arms retain the same native algorithms, including atomic embedding accumulation, and request no determinism. Both disable redundant uninitialized fills. The seven immutable V4 deterministic-gradient full-training pairs isolate update equivalence and remain a prerequisite; no new deterministic comparison replaces the original native baseline.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(jobs=len(jobs),source_sha256=frozen['source_sha256'],protocol_sha256=digest(output))))

if __name__=='__main__':
    main()
