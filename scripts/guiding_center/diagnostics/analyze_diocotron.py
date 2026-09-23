"""Audit disk diocotron growth, positivity, invariants and solver cost from logs.

The analytical dispersion relation is for a sharp annulus. Agreement for the
paper's steep profile is an approximation and requires spatial/profile checks.
No simulations are launched by this module.
"""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.guiding_center.diagnostics.diocotron_reference import annulus_spectrum, converged_smooth_reference, fit_growth, PAPER_FIT_WINDOW


def read_run(path):
    path=Path(path)
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or rows[0].get("step")!=0:
        raise ValueError(f"{path}: an initial diagnostics row is required")
    times=np.array([row["time"] for row in rows],dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError(f"{path}: times must be finite and strictly increasing")
    config=rows[0].get("run_configuration")
    if config is None or config.get("case")!="diocotron_k":
        raise ValueError(f"{path}: diocotron run_configuration metadata is missing")
    timing_path=path.with_name(path.stem+"_timings.jsonl")
    timings=([json.loads(line) for line in timing_path.read_text().splitlines() if line.strip()]
             if timing_path.exists() else [])
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows=rows,times=times,config=config,timings=timings)


def column(rows,key):
    return np.array([r.get(key) if r.get(key) is not None else np.nan for r in rows],dtype=float)


def invariant_summary(rows,key):
    values=column(rows,key)
    if not np.isfinite(values).all():
        return dict(status="missing_or_nonfinite")
    difference=values-values[0]
    result=dict(status="measured",initial=float(values[0]),final=float(values[-1]),
                final_drift=float(difference[-1]),max_absolute_drift=float(np.max(np.abs(difference))))
    if values[0]:
        result.update(final_relative_drift=float(difference[-1]/values[0]),
                      max_relative_drift=float(np.max(np.abs(difference))/abs(values[0])))
    return result


def _match_runs(run,other,*,different_dt=False,unperturbed=False):
    """Require a matched physical/spatial case; never compare unrelated curves."""
    left,right=run["config"],other["config"]
    keys=("mesh_size","triangles","order","time_scheme","volume_quadrature","volume_quad_1d","edge_quad_1d","initial_projection_quad_1d","poisson_tau_initial")
    if not different_dt:
        keys+=("dt",)
    errors=[key for key in keys if left.get(key)!=right.get(key)]
    excluded={"epsilon","eps"}
    if unperturbed:
        excluded|={"k","modes"}
    parameters_left={k:v for k,v in left["case_parameters"].items() if k not in excluded}
    parameters_right={k:v for k,v in right["case_parameters"].items() if k not in excluded}
    if parameters_left!=parameters_right:
        errors.append("case_parameters")
    if errors:
        raise ValueError("comparison configuration differs: "+", ".join(errors))


def compare_control(run,control,key,window):
    _match_runs(run,control,unperturbed=True)
    if control["config"]["case_parameters"].get("epsilon")!=0:
        raise ValueError("the unperturbed control must have epsilon=0")
    selected=(run["times"]>=window[0]-1.e-10)&(run["times"]<=window[1]+1.e-10)
    if not np.any(selected) or control["times"][-1]<window[1]-1.e-10:
        return dict(status="incomplete_window")
    # Matching dt/cadence is checked using time keys; no scalar-amplitude
    # subtraction is used as a substitute for subtracting complex fields.
    lookup={round(row["time"],10):row.get(key) for row in control["rows"]}
    fractions=[]
    for row,active in zip(run["rows"],selected):
        if not active:
            continue
        noise=lookup.get(round(row["time"],10))
        amplitude=row.get(key)
        if noise is None or amplitude is None or not np.isfinite([noise,amplitude]).all() or amplitude<=0:
            return dict(status="missing_or_invalid_matched_samples")
        fractions.append(noise/amplitude)
    return dict(status="measured",max_control_fraction=float(max(fractions)),samples=len(fractions))


def positivity_summary(run):
    endpoints=run["rows"]
    stages=[entry for row in run["timings"] for entry in row.get("positivity_stage_checks",[])]
    all_checks=[*endpoints,*stages]
    if any(row.get("rho_min_checked") is None for row in endpoints):
        return dict(status="not_measured")
    minima=column(all_checks,"rho_min_checked")
    if not np.isfinite(minima).all():
        return dict(status="nonfinite")
    tolerances=column(all_checks,"positivity_tolerance")
    violated=minima < -tolerances
    first=next((row for row,bad in zip(all_checks,violated) if bad),None)
    # Endpoints appear before internal stages: choose the earliest time witness.
    if first is not None:
        first=min((r for r,bad in zip(all_checks,violated) if bad),key=lambda r:r["time"])
    last_step=int(endpoints[-1]["step"])
    stage_labels={int(row["step"]):{r.get("stage") for r in row.get("positivity_stage_checks",[])}
                  for row in run["timings"] if row.get("step",0)>0}
    required={"ARK stage 2","ARK stage 3","ARK stage 4"}
    full_stages=(run["config"]["time_scheme"]=="imex-ark3" and all(
        required.issubset(stage_labels.get(step,set())) for step in range(1,last_step+1)))
    all_bounds=np.all(column(all_checks,"rho_bernstein_lower_bound") >= -tolerances)
    status="violated" if violated.any() else "bound_satisfied" if all_bounds and full_stages else "inconclusive"
    return dict(status=status,initial_status=endpoints[0].get("positivity_status"),
        initial_minimum=float(minima[0]),minimum=float(min(minima)),
        minimum_cell_average=float(np.nanmin(column(all_checks,"rho_cell_average_min"))),
        maximum_negative_mass_quadrature=float(np.nanmax(column(all_checks,"rho_negative_mass_quadrature"))),
        initial_negative_mass_quadrature=endpoints[0].get("rho_negative_mass_quadrature"),
        stage_checks=len(stages),all_ark_stages_available=full_stages,
        first_negative_witness=None if first is None else dict(time=first["time"],stage=first.get("stage","initial" if first.get("step")==0 else "endpoint"),minimum=first["rho_min_checked"]),
        interpretation="Negative samples prove a violation. Negative Bernstein bounds alone are inconclusive; negative-part integrals use volume quadrature.")


def tau_summary(run):
    rows=run["timings"] or run["rows"]
    events=[event for row in rows for event in row.get("poisson_tau_retry_events",[])]
    values=[float(r["poisson_tau"]) for r in rows if r.get("poisson_tau") is not None]
    values.extend(float(event[key]) for event in events for key in ("tau_before","tau_after") if key in event)
    return dict(retries=len(events),values=sorted(set(values)),fixed=(len(events)==0 and len(set(values))<=1))


def analyze(run,*,control=None,half_seed=None,refined=None,window=PAPER_FIT_WINDOW,rate_tolerance=.05):
    if not np.isfinite(rate_tolerance) or rate_tolerance <= 0:
        raise ValueError("rate tolerance must be finite and positive")
    parameters=run["config"]["case_parameters"]
    mode=int(parameters["k"])
    sharp_reference=annulus_spectrum([mode],inner=parameters["s_minus"],outer=parameters["s_plus"],density=parameters["rho_bar"])[0]
    smooth=((parameters.get("p") is not None and not parameters.get("truncate",False)) or
            (parameters.get("p") is None and parameters.get("edge_width",0)>0))
    reference=(converged_smooth_reference(mode,parameters=parameters) if smooth else
               dict(**sharp_reference,reference_type="sharp-annulus-analytic"))
    gamma=reference["growth_rate"]
    amplitude_key=f"diocotron_phi_mode_{mode}_l2"
    fit=lambda data,key,interval: fit_growth(data["times"],column(data["rows"],key),window=interval,reference=gamma if gamma>0 else None)
    target=fit(run,amplitude_key,window)
    mid=sum(window)/2
    growth=dict(reference=reference,sharp_reference=sharp_reference,mode=mode,window=list(window),
        target_mode=target,paper_potential_norm=fit(run,"diocotron_phi_eq_l2",window),
        subwindows=[fit(run,amplitude_key,(window[0],mid)),fit(run,amplitude_key,(mid,window[1]))],
        relative_rate_tolerance=rate_tolerance,
        agrees_with_reference=(gamma>0 and target.get("status")=="fitted" and
                                    abs(target["relative_error"])<=rate_tolerance))
    sharp_gamma=sharp_reference["growth_rate"]
    growth["sharp_annulus_relative_error"]=(
        (target["growth_rate"]-sharp_gamma)/sharp_gamma
        if target.get("status")=="fitted" and sharp_gamma>0 else None)
    # Keep the article's sharp-layer prediction visible for smoothed cases.
    # A smoothing discrepancy is a model difference, not solely timestep error.
    notes=[]
    if gamma==0:
        notes.append("The selected sharp-annulus mode is stable; a positive exponential growth rate is not expected.")
    if smooth:
        if not reference["reference_converged"]:
            notes.append("The smooth radial eigenvalue reference has not converged under radial refinement.")
        notes.append("The smooth-profile rate is an independent numerical eigenvalue reference; an HDG spatial refinement check remains necessary.")
    elif parameters.get("p") is not None or parameters.get("edge_width",0):
        notes.append("Sharp-annulus theory is approximate for this radial profile; spatial and profile refinement remain necessary.")
    if not growth["agrees_with_reference"]:
        notes.append("The target-mode fit has not demonstrated agreement with the selected equilibrium reference.")
    if target.get("r_squared",0) is None or target.get("r_squared",0)<.995:
        notes.append("The target-mode curve has not established a clean exponential window (R² >= 0.995).")
    for subfit in growth["subwindows"]:
        if subfit.get("status")!="fitted" or (gamma and abs(subfit["growth_rate"]-target.get("growth_rate",0))/gamma>.05):
            notes.append("Growth is not consistent across both halves of the fitting window.")
            break
    tau=tau_summary(run)
    if not tau["fixed"]:
        notes.append("Poisson tau changed; this run does not provide a fixed-tau rate comparison.")
    selected=(run["times"]>=window[0]-1.e-10)&(run["times"]<=window[1]+1.e-10)
    ratios=column(run["rows"],"diocotron_phi_harmonic_ratio")[selected]
    growth["maximum_potential_harmonic_ratio_in_window"]=(float(max(ratios)) if len(ratios) and np.isfinite(ratios).all() else None)
    if not len(ratios) or not np.isfinite(ratios).all() or max(ratios)>.1:
        notes.append("A weak second harmonic (2m/m <= 0.1) has not been established throughout the window.")
    if control is None:
        notes.append("An unperturbed control is needed to bound mesh/equilibrium noise.")
    else:
        growth["control"]=compare_control(run,control,amplitude_key,window)
        if growth["control"].get("max_control_fraction",np.inf)>.1:
            notes.append("Control noise has not been shown to remain below 10% of the seeded mode.")
        if not tau_summary(control)["fixed"] or tau_summary(control)["values"]!=tau["values"]:
            notes.append("Control and perturbed run did not retain the same fixed Poisson tau.")
    if half_seed is None:
        notes.append("A half-amplitude seed run is needed to check linearity.")
    else:
        _match_runs(run,half_seed)
        if not np.isclose(half_seed["config"]["case_parameters"]["epsilon"],parameters["epsilon"]/2,rtol=1.e-12,atol=0):
            raise ValueError("half-seed run must use half the perturbation amplitude")
        half_fit=fit(half_seed,amplitude_key,window)
        growth["half_seed"]=half_fit
        if half_fit.get("status")!="fitted" or gamma==0 or abs(half_fit["growth_rate"]-target.get("growth_rate",0))/gamma>.05:
            notes.append("Seed-independent growth has not been established.")
        lookup={round(r["time"],10):r.get(amplitude_key) for r in half_seed["rows"]}
        deviations=[]
        for row,active in zip(run["rows"],selected):
            if active:
                half=lookup.get(round(row["time"],10)); full=row.get(amplitude_key)
                deviations.append(abs(2*half/full-1) if half is not None and full is not None and full>0 else np.inf)
        maximum=max(deviations,default=np.inf)
        growth["half_seed_amplitude_scaling_max_error"]=float(maximum) if np.isfinite(maximum) else None
        if maximum>.1:
            notes.append("Doubling the half-seed mode does not agree within 10% throughout the window.")
        if not tau_summary(half_seed)["fixed"] or tau_summary(half_seed)["values"]!=tau["values"]:
            notes.append("Half-seed and main run did not retain the same fixed Poisson tau.")
    if refined is None:
        notes.append("A smaller timestep is needed to assess temporal sensitivity.")
    else:
        _match_runs(run,refined,different_dt=True)
        if refined["config"]["dt"]>=run["config"]["dt"] or refined["config"]["case_parameters"]["epsilon"]!=parameters["epsilon"]:
            raise ValueError("refined run must use the same seed and a smaller dt")
        refined_fit=fit(refined,amplitude_key,window)
        growth["refined_dt"]=refined_fit
        if refined_fit.get("status")!="fitted" or gamma==0 or abs(refined_fit["growth_rate"]-target.get("growth_rate",0))/gamma>.02:
            notes.append("Growth-rate change under timestep refinement has not been shown below 2%.")
        if not tau_summary(refined)["fixed"] or tau_summary(refined)["values"]!=tau["values"]:
            notes.append("Refined and main run did not retain the same fixed Poisson tau.")
    # A high-mode run can be contaminated by neighboring unstable modes even
    # when the target's second harmonic is small. Use the full recorded band.
    competing=[]
    for key in run["rows"][0]:
        if key.startswith("diocotron_phi_mode_") and key.endswith("_l2"):
            number=key[len("diocotron_phi_mode_"):-len("_l2")]
            if number.isdigit() and int(number) not in (mode,2*mode,3*mode):
                competing.append((int(number),key))
    candidates=[]
    for row,active in zip(run["rows"],selected):
        target_value=row.get(amplitude_key)
        if active and target_value is not None and np.isfinite(target_value) and target_value>0:
            for number,key in competing:
                value=row.get(key)
                if value is not None and np.isfinite(value):
                    candidates.append((value/target_value,number,row["time"]))
    if candidates:
        ratio,number,time=max(candidates)
        growth["maximum_competing_mode_fraction"]=dict(ratio=float(ratio),mode=number,time=time,
            recorded_modes=len(competing),interpretation="Largest recorded nonharmonic potential mode / target mode in the fixed fit window; not a substitute for an unperturbed control.")
        if ratio>.1:
            notes.append("A competing nonharmonic mode exceeds 10% of the seeded mode in the fitting window.")
    positive=positivity_summary(run)
    costs=run["timings"][1:]
    return dict(source="https://arxiv.org/abs/1909.05005",input=dict(path=run["path"],sha256=run["sha256"]),
        comparisons={name:dict(path=data["path"],sha256=data["sha256"]) for name,data in
                     (("control",control),("half_seed",half_seed),("refined",refined)) if data is not None},
        configuration=run["config"],final_time=float(run["times"][-1]),growth=growth,
        positivity=positive,invariants={key:invariant_summary(run["rows"],key) for key in ("mass","energy_from_q_l2","enstrophy")},
        poisson_tau=tau,qualification_notes=notes,
        cost={key:float(np.nansum(column(costs,key))) for key in
              ("linear_step_wall_time","transport_time","poisson_time","explicit_residual_time","positivity_stage_time","transport_stage_count","poisson_stage_count")})


def write_report(summary,run,output,*,plots=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    reference=summary["growth"]["reference"]
    text=["# Disk diocotron benchmark audit", "",
        "Reference: [Zoni & Güçlü, §6.3, equations (33)–(35)](https://arxiv.org/abs/1909.05005).", "",
        f"Mode {reference['mode']}; {reference['reference_type']} γ = {reference['growth_rate']:.10f}, Re(ω) = {reference['omega_real']:.10f}.",
        f"Data end at t={summary['final_time']:.6g}; fixed fit window {summary['growth']['window']}.", "",
        "| Observable | Fit status | Growth rate | Relative error | R² |", "|---|---|---:|---:|---:|"]
    for name,key in (("Target potential mode","target_mode"),("Paper whole potential norm","paper_potential_norm")):
        fit=summary["growth"][key]
        text.append(f"| {name} | {fit['status']} | {fit.get('growth_rate','—')} | {fit.get('relative_error','—')} | {fit.get('r_squared','—')} |")
    sharp=summary["growth"]["sharp_reference"]
    if reference["reference_type"] != "sharp-annulus-analytic":
        text += ["", f"Zoni–Güçlü equation (34), evaluated for these annulus radii: sharp-layer γ = {sharp['growth_rate']:.10f}, Re(ω) = {sharp['omega_real']:.10f}. "
                 f"Measured target-mode relative difference from that sharp-layer rate: {summary['growth']['sharp_annulus_relative_error']}. "
                 "This is a separate comparison: the smooth radial profile has a different equilibrium spectrum."]
    text += ["",f"Positivity: **{summary['positivity']['status']}**. See summary.json for initialization, internal stages, bounds, and negative-part quadrature.","",
             "| Invariant | Final relative drift | Maximum relative drift |", "|---|---:|---:|"]
    for key,value in summary["invariants"].items():
        text.append(f"| {key} | {value.get('final_relative_drift','—')} | {value.get('max_relative_drift','—')} |")
    text += ["", "Energy is ½∫|q_h|² and enstrophy is ½∫ρ_h². Conservation drift is measured from the numerical initial state; it does not include projection error.", "",
             f"Poisson τ audit: `{summary['poisson_tau']}`.", "", "## Remaining qualification checks", ""]
    text += ["- "+note for note in summary["qualification_notes"]]
    text += ["", "Regression standard errors describe the fit, not discretization error. Positivity diagnostics do not apply a limiter. Polar norms cover the recorded inscribed-circle radius; the paper norm uses the full polygon.", ""]
    (output/"report.md").write_text("\n".join(text))
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t=run["times"];rows=run["rows"]
        fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
        ax=axes[0,0]
        for key,label in (("diocotron_phi_eq_l2","whole potential"),("diocotron_phi_mode_target_l2","target mode"),("diocotron_phi_axisymmetric_l2","axisymmetric")):
            y=column(rows,key);mask=np.isfinite(y)&(y>0)
            ax.semilogy(t[mask],y[mask],label=label)
        fit=summary["growth"]["target_mode"]
        if fit["status"]=="fitted":
            x=np.array(summary["growth"]["window"])
            ax.semilogy(x,np.exp(fit["log_intercept"]+fit["growth_rate"]*x),"k--",label="measured fit")
            anchor=float(np.mean(x));intercept=fit["log_intercept"]+(fit["growth_rate"]-reference["growth_rate"])*anchor
            ax.semilogy(x,np.exp(intercept+reference["growth_rate"]*x),":",color="red",label="equilibrium reference slope")
        ax.set(xlabel="time",ylabel="potential perturbation L2");ax.legend()
        for key,label in (("mass","mass"),("energy_from_q_l2","energy"),("enstrophy","enstrophy")):
            y=column(rows,key)
            if np.isfinite(y).all() and y[0]: axes[0,1].plot(t,(y-y[0])/abs(y[0]),label=label)
        axes[0,1].set(xlabel="time",ylabel="relative invariant drift");axes[0,1].legend()
        axes[1,0].plot(t,column(rows,"rho_min_checked"),label="checked endpoint minimum")
        stages=[r for row in run["timings"] for r in row.get("positivity_stage_checks",[])]
        if stages: axes[1,0].scatter(column(stages,"time"),column(stages,"rho_min_checked"),s=3,label="internal/endpoint stages")
        axes[1,0].axhline(0,color="k",lw=.5);axes[1,0].set(xlabel="time",ylabel="density minimum");axes[1,0].legend()
        mode=summary["growth"]["mode"]
        shown=set(range(1,17)) | {m for m in (mode-2,mode-1,mode,mode+1,mode+2,2*mode,3*mode) if m>0}
        competitor=summary["growth"].get("maximum_competing_mode_fraction")
        if competitor is not None:
            shown.add(competitor["mode"])
        for m in sorted(shown):
            key=f"diocotron_phi_mode_{m}_l2"
            y=column(rows,key);mask=np.isfinite(y)&(y>0)
            if np.any(mask): axes[1,1].semilogy(t[mask],y[mask],label=str(m))
        axes[1,1].set(xlabel="time",ylabel="potential mode L2");axes[1,1].legend(title="mode",ncol=4,fontsize=7)
        fig.savefig(output/"diagnostics.png",dpi=170);fig.savefig(output/"diagnostics.pdf");plt.close(fig)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path,help="Physics diagnostics JSONL, including t=0.")
    parser.add_argument("--control",type=Path)
    parser.add_argument("--half-seed",type=Path)
    parser.add_argument("--refined",type=Path)
    parser.add_argument("--fit-window",type=float,nargs=2,default=PAPER_FIT_WINDOW)
    parser.add_argument("--rate-tolerance",type=float,default=.05)
    parser.add_argument("--output-dir",type=Path,default=Path("artifacts/diocotron_analysis"))
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    run=read_run(args.run)
    summary=analyze(run,control=None if args.control is None else read_run(args.control),
                    half_seed=None if args.half_seed is None else read_run(args.half_seed),
                    refined=None if args.refined is None else read_run(args.refined),
                    window=tuple(args.fit_window),rate_tolerance=args.rate_tolerance)
    write_report(summary,run,args.output_dir,plots=not args.no_plots)
    print((args.output_dir/"report.md").read_text())


if __name__=="__main__":
    main()
