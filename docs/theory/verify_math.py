#!/usr/bin/env python3
"""NumPy-only theory/CSV verification. Never imports DXY-MID or measures an image."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = 'ABCD'


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return float(row[key]) if row.get(key, '') else math.nan


def flag(row, key):
    return row[key].lower() in ('true', '1')


def close(actual, expected, *, atol=2e-10, rtol=2e-11):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)


def rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def four_lengths(a, b, body, direction, length, diameter=1.0, theta=0.0):
    """Independent published equations; body columns are native-image half axes."""
    T = np.diag([1.0, a / b]) @ rotation(theta).T
    A = length * diameter / (2 * a)
    return np.array([A, A * np.sqrt(a / b),
                     A * np.linalg.norm(T @ direction),
                     2 * np.linalg.svd(T @ body, compute_uv=False)[0] * diameter / (2 * a)])


def spectral_formula(c, d, rho, alpha):
    g = np.cos(alpha)**2 + rho**2 * np.sin(alpha)**2
    tau = c*c*g + d*d*(np.sin(alpha)**2 + rho**2*np.cos(alpha)**2)
    determinant = rho**2 * c*c * d*d
    return np.sqrt((tau + np.sqrt(max(0.0, tau*tau - 4*determinant))) / 2)


def check_algebra():
    rng = np.random.default_rng(824711)
    worst_spectral = 0.0
    worst_gradient = 0.0
    for _ in range(1000):
        a = rng.uniform(10, 150)
        b = a * rng.uniform(.25, 1)
        c = rng.uniform(5, 80)
        d = c * rng.uniform(.25, 1)
        phi, theta = rng.uniform(-1.5, 1.5, 2)
        U = rotation(theta + phi)
        body = U @ np.diag([c, d])
        values = four_lengths(a, b, body, U[:, 0], 2*c, theta=theta)
        A, B, C, D = values
        assert B >= A-1e-10 and D >= C-1e-10 and C >= A-1e-10
        close(four_lengths(a, a, body, U[:, 0], 2*c, theta=theta), np.full(4, c/a))
        close(B/A, np.sqrt(a/b))
        close(B, (2*c)/(2*np.sqrt(a*b)))  # determinant/area-equivalent scale
        close(C/A, np.sqrt(np.cos(phi)**2 + (a/b)**2*np.sin(phi)**2))
        analytic = spectral_formula(c, d, a/b, phi) / a
        worst_spectral = max(worst_spectral, abs(analytic-D))
        close(analytic, D)
        circle = U * c
        close(four_lengths(a, b, circle, U[:, 0], 2*c, theta=theta)[3]/A, a/b)
        parallel = rotation(theta)
        close(four_lengths(a, b, parallel @ np.diag([c,d]), parallel[:,0], 2*c, theta=theta)[2], A)
        close(four_lengths(a, b, body, parallel[:,1], 2*c, theta=theta)[2]/A, a/b)
        close(four_lengths(a, b, parallel @ np.diag([c,d]), parallel[:,0], 2*c, theta=theta)[3]/A, max(1, (a/b)*(d/c)))
        # Finite log propagation for arbitrary positive components.
        L1, a1, h0, h1 = rng.uniform(.5, 2, 4)
        close(np.log((L1/a1)*h1/h0), np.log(L1)-np.log(a1)+np.log(h1/h0))
        # C derivative: fixed L,a,b except local relative angle (radians).
        rho = a/b
        g = np.cos(phi)**2 + rho*rho*np.sin(phi)**2
        grad = (rho*rho-1)*np.sin(phi)*np.cos(phi)/g
        eps = 1e-6
        logh = lambda p: .5*np.log(np.cos(p)**2+rho*rho*np.sin(p)**2)
        numerical = (logh(phi+eps)-logh(phi-eps))/(2*eps)
        worst_gradient = max(worst_gradient, abs(grad-numerical))
        close(numerical, grad, atol=1e-8)
    # Explicit counterexamples to a universal B-vs-C/D ordering.
    thin_parallel=four_lengths(2.,1.,np.diag([3.,.2]),np.array([1.,0.]),6.)
    circle=four_lengths(2.,1.,3*np.eye(2),np.array([1.,0.]),6.)
    thin_perpendicular=four_lengths(2.,1.,rotation(np.pi/2)@np.diag([3.,.2]),np.array([0.,1.]),6.)
    assert thin_parallel[1]>thin_parallel[2] and thin_parallel[1]>thin_parallel[3]
    assert circle[3]>circle[1] and thin_perpendicular[2]>thin_perpendicular[1]
    return dict(random_cases=1000, spectral_formula_max_mm_difference=worst_spectral,
                C_gradient_max_abs_difference=worst_gradient)


def check_data(data):
    rows = read_csv(data/'perturbation_trials.csv')
    scenes = read_csv(data/'scenes.csv')
    audit = read_csv(data/'matching_audit.csv')
    n0 = {r['scene_id']: r for r in rows if r['cohort']=='N0'}
    assert len(rows)==927 and len(n0)==len(scenes)==len(audit)==9
    assert len({r['model'] for r in scenes})==3
    ids = set(n0)
    assert ids == {r['scene_id'] for r in scenes} == {r['scene_id'] for r in audit}
    pass_keys = [k for k in audit[0] if k.endswith('_pass') and k!='joint_pass']
    assert len(pass_keys)==8 and all(all(flag(r,k) for k in pass_keys+['joint_pass']) for r in audit)
    original_audit=read_csv(data/'selection_audit_30.csv')
    assert len(original_audit)==30 and len({r['scene_id'] for r in original_audit})==30
    assert all(all(flag(r,k) for k in pass_keys)==flag(r,'joint_pass') for r in original_audit)
    assert {r['scene_id'] for r in original_audit if flag(r,'joint_pass')}==ids

    assert all(flag(r,'original_joint_pass') and flag(r,'recomputed_joint_pass') for r in scenes)
    assert len({r['input_id'] for r in rows})==927
    frozen_gt = {r['scene_id']: number(r,'GT_mm') for r in scenes}
    same_n0 = read_csv(data/'n0_results.csv')
    assert same_n0 == [r for r in rows if r['cohort']=='N0']
    variance_values = {}
    max_log_residual = 0.0
    failures = 0
    for r in rows:
        sid = r['scene_id']
        close(number(r,'GT_mm'), frozen_gt[sid])
        published = np.array([number(r,f'D_{m}_mm') for m in METHODS])
        finite = np.isfinite(published)
        assert finite.all() or not finite.any(), r['input_id']
        if not finite.all():
            failures += 1
            assert not flag(r,'measurement_eligible')
            continue
        a,b,L = [number(r,k) for k in ['ring_a_px','ring_b_px','L_px']]
        assert a>=b>0 and L>0
        close(number(r,'q'),b/a)
        close(number(r,'D_mid_mm'),number(r,'ring_inner_mm')+1.2)
        body = np.array([[number(r,f'body_basis_{i}{j}_px') for j in range(2)] for i in range(2)])
        direction = np.array([number(r,'axis_x2')-number(r,'axis_x1'),number(r,'axis_y2')-number(r,'axis_y1')])
        close(np.linalg.norm(direction), L)
        sv = np.linalg.svd(body, compute_uv=False)
        close(2*sv[0],L)
        close(sv,[number(r,'body_a_px'),number(r,'body_b_px')])
        unit=direction/np.linalg.norm(direction)
        close(body@body.T@unit,sv[0]**2*unit,atol=5e-7,rtol=5e-10)
        close([(number(r,'axis_x1')+number(r,'axis_x2'))/2,(number(r,'axis_y1')+number(r,'axis_y2'))/2],
              [number(r,'body_center_x_px'),number(r,'body_center_y_px')])
        ring_direction=rotation(np.deg2rad(number(r,'ring_angle_deg')))[:,0]
        phi=np.rad2deg(np.arccos(np.clip(abs(unit@ring_direction),0,1)))
        close(phi,number(r,'phi_deg'),atol=1e-8)

        reconstructed = four_lengths(a,b,body,direction/np.linalg.norm(direction),L,number(r,'D_mid_mm'),np.deg2rad(number(r,'ring_angle_deg')))
        close(reconstructed,published)
        assert published[3]>=published[2]-1e-9 and published[2]>=published[0]-1e-9 and published[1]>=published[0]-1e-9
        if r['chosen_shape']=='circle':close(published[3]/published[0],a/b)
        ref = n0[sid]
        for i,m in enumerate(METHODS):
            h = published[i]/published[0]
            close(h,number(r,f'h_{m}'))
            close(number(r,f'N0_{m}_mm'),number(ref,f'D_{m}_mm'))
            delta = 100*(published[i]/number(ref,f'D_{m}_mm')-1)
            signed = 100*(published[i]/number(r,'GT_mm')-1)
            close(delta,number(r,f'sensitivity_{m}_pct'))
            close(signed,number(r,f'signed_{m}_pct'))
            close(abs(signed),number(r,f'ARE_{m}_pct'))
            logL = np.log(L/number(ref,'L_px'))
            minusa = -np.log(a/number(ref,'ring_a_px'))
            logh = np.log(h/number(ref,f'h_{m}'))
            residual = np.log(published[i]/number(ref,f'D_{m}_mm'))-logL-minusa-logh
            max_log_residual = max(max_log_residual,abs(residual))
            close(residual,0,atol=1e-12)
            if r['cohort']=='random20':variance_values.setdefault((sid,r['condition'],m),[]).append([logL,minusa,logh])
    # Metrics are computed from public N0 values, not copied from the report.
    calculated=[]
    for m in METHODS:
        values = np.array([number(r,f'D_{m}_mm') for r in n0.values()])
        gt = np.array([number(r,'GT_mm') for r in n0.values()])
        err = values-gt
        rel = 100*err/gt
        referr = abs(100*(np.array([number(r,'D_A_mm') for r in n0.values()])-gt)/gt)
        result=dict(method=m,n_scenes=9,n_source_models=3,median_ARE_pct=np.median(abs(rel)),MAE_mm=np.mean(abs(err)),mean_ARE_pct=np.mean(abs(rel)),median_signed_pct=np.median(rel),p95_ARE_pct=np.quantile(abs(rel),.95),max_ARE_pct=max(abs(rel)),improved_vs_A=int((abs(rel)<referr-1e-9).sum()))
        expected = next(r for r in read_csv(data/'n0_metrics.csv') if r['method']==m)
        for k,v in result.items():
            if k!='method':close(v,number(expected,k))
        calculated.append(result)
    assert calculated[2]['improved_vs_A']==8
    random_rows = [r for r in rows if r['cohort']=='random20']
    assert len(random_rows)==720
    seeds = set(str(i) for i in range(2026102001,2026102021))
    for sid in ids:
        for cond in ['N1','N2','N3','N4']:
            g = [r for r in random_rows if r['scene_id']==sid and r['condition']==cond]
            assert len(g)==20 and {r['repeat_seed'] for r in g}==seeds
            for r in g:
                fp,fr=number(r,'polyp_factor'),number(r,'ring_factor')
                assert (1.05<=fp<=1.10 or fp==1) and (1.05<=fr<=1.10 or fr==1)
                if cond=='N1':assert fr==1
                if cond=='N2':assert fp==1
                if cond=='N4':assert fp==fr
    summaries=read_csv(data/'prediction_distributions.csv')
    assert len(summaries)==144
    for r in summaries:
        g=[x for x in random_rows if x['scene_id']==r['scene_id'] and x['condition']==r['condition']]
        x=r['method'];v=np.array([number(z,f'D_{x}_mm') for z in g]);valid=np.isfinite(v);v=v[valid]
        gt=number(n0[r['scene_id']],'GT_mm');base=number(n0[r['scene_id']],f'D_{x}_mm')
        shift=100*(v/base-1);signed=100*(v/gt-1)
        values=dict(N0_mm=base,n_repeats=20,n_numeric=len(v),numeric_failure_frequency=1-len(v)/20,
            prediction_median_mm=np.median(v),prediction_mean_mm=np.mean(v),prediction_std_mm=np.std(v,ddof=1),empirical_p05_mm=np.quantile(v,.05),empirical_p95_mm=np.quantile(v,.95),median_change_from_N0_pct=100*(np.median(v)/base-1),mean_signed_error_pct=np.mean(signed),median_signed_error_pct=np.median(signed),median_absolute_relative_error_pct=np.median(abs(signed)),MAE_mm=np.mean(abs(v-gt)),p95_absolute_relative_error_pct=np.quantile(abs(signed),.95),max_absolute_relative_error_pct=max(abs(signed)),max_abs_departure_from_N0_pct=max(abs(shift)),max_abs_departure_from_N0_mm=max(abs(v-base)))
        for key,col in [('candidate_switch_frequency','candidate_switch'),('shape_switch_frequency','shape_switch'),('body_switch_frequency','body_switch'),('parameter_jump_frequency','parameter_jump'),('same_identity_parameter_jump_frequency','same_identity_parameter_jump'),('Gate_change_frequency','Gate_change'),('Q_recovery_change_frequency','Q_conflict_recovery_change')]:values[key]=sum(flag(z,col) for z in g)/20
        for k,v in values.items():close(v,number(r,k))
    for r in read_csv(data/'N4_cancellation.csv'):
        g=[z for z in random_rows if z['scene_id']==r['scene_id'] and z['condition']=='N4']
        change=np.array([number(z,'sensitivity_A_pct') for z in g])
        for key,value in dict(n=20,n_numeric=len(change),median_change_A_pct=np.median(change),p05_change_A_pct=np.quantile(change,.05),p95_change_A_pct=np.quantile(change,.95),max_abs_change_A_pct=max(abs(change)),candidate_switches=sum(flag(z,'candidate_switch') for z in g),parameter_jumps=sum(flag(z,'parameter_jump') for z in g)).items():close(value,number(r,key))
    fixed=[r for r in rows if r['cohort']=='fixed']
    assert len(fixed)==162
    for sid in ids:
        for cond in ['N1','N2','N3','N4']:
            g=[r for r in fixed if r['scene_id']==sid and r['condition']==cond]
            factors={1.05,1.075,1.10}
            if cond=='N3':assert {(number(r,'polyp_factor'),number(r,'ring_factor')) for r in g}=={(p,q) for p in factors for q in factors}
            else:
                assert len(g)==3
                for r in g:
                    p,q=number(r,'polyp_factor'),number(r,'ring_factor')
                    if cond=='N1':assert p in factors and q==1
                    if cond=='N2':assert p==1 and q in factors
                    if cond=='N4':assert p==q and p in factors
    for r in read_csv(data/'log_variance_decomposition.csv'):
        z=np.array(variance_values[(r['scene_id'],r['condition'],r['method'])]);cov=np.cov(z,rowvar=False,ddof=1)
        actual=np.var(z.sum(axis=1),ddof=1)
        terms=[cov[0,0],cov[1,1],cov[2,2],2*cov[0,1],2*cov[0,2],2*cov[1,2]]
        close(terms,[number(r,k) for k in ['var_log_L','var_minus_log_a','var_log_h','twice_cov_L_minus_a','twice_cov_L_h','twice_cov_minus_a_h']])
        close(sum(terms),actual);close(actual,number(r,'actual_var_log_D'))
    # Same geometry/NaN pattern for all methods; these are numerical successes only.
    return dict(N0_metrics=calculated,scenes=9,source_models=3,trial_rows=len(rows),random_rows=720,numeric_failures=failures,distribution_rows_checked=144,propagation_max_abs_log_residual=max_log_residual,geometry_or_image_recomputed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',type=Path,help='Optional result file; inputs are always read-only')
    args=parser.parse_args()
    math_results=check_algebra()
    results=check_data(ROOT/'data')
    manifest=ROOT/'PUBLICATION_MANIFEST.json'
    if manifest.exists():
        for name,expected in json.loads(manifest.read_text())['files_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    out=dict(status='PASS',algebra=math_results,data=results,scope='mathematics and published CSV recomputation only; no original scene end-to-end measurement')
    output=json.dumps(out,ensure_ascii=False,indent=2)
    if args.json:args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(output+'\n')
    print(output)


if __name__=='__main__':main()
