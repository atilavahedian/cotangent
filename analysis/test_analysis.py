"""Verify endpoint and censoring rules independently of training code."""
import math
import pytest
from analyze import crossing, interval, paired


def test_first_scheduled_crossing_and_missing_target():
    curve = [{"step":100,"validation_bpb":3.3,"elapsed_seconds":2.0},
             {"step":200,"validation_bpb":3.19,"elapsed_seconds":4.2},
             {"step":300,"validation_bpb":3.1,"elapsed_seconds":6.0}]
    assert crossing(curve,3.2) is curve[1]
    assert crossing(curve,3.0) is None


def test_paired_success_requires_time_and_quality_with_no_censor_imputation():
    rows=[]
    for seed in range(5):
        for mode in ("native","adaptive"):
            rows.append({"group":"small","seed":seed,"mode":mode,
                         "target_seconds":10 if mode=="native" else 8,
                         "training_seconds":20 if mode=="native" else 18,
                         "test_bpb":2.5 if mode=="native" else 2.51})
    frozen={"quality_noninferiority_margin_bpb":.03,"meaningful_time_reduction_fraction":.1}
    out=paired(rows,"small",frozen)
    assert out["success"]
    assert out["elapsed_reduction_fraction"]["mean"]==pytest.approx(.2)
    rows[-1]["target_seconds"]=None
    out=paired(rows,"small",frozen)
    assert not out["success"] and not out["time_gate_pass"]
    assert out["elapsed_reduction_fraction"] is None
    assert out["pairs"][-1]["elapsed_reduction_fraction"] is None
    assert out["quality_gate_pass"]


def test_t_interval_uses_seed_variation_and_rejects_unplanned_n():
    values=[1,2,3,4,5]
    ci=interval(values)
    assert ci["mean"]==3
    assert ci["upper"]==pytest.approx(3+2.7764451051977987*math.sqrt(2.5/5))
    with pytest.raises(ValueError):
        interval([1,2])
