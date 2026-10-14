"""Compare two trained DNNs with their corresponding classical/numerical PSA models.

Runs the comparison for 100 input configurations and saves the results.
The same uniformly sampled inputs are used by all four calculations.
Errors in histograms are signed DNN minus classical, in dB.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
from psa_nf_gain import compute_nf_and_maxgain
from predict import predict_nf_gain

BASE_DIR = Path(__file__).resolve().parent
FIELDS = ['case', 'pump_power_dBm', 'pump_signal_sep_nm', 'beta2_ps2_per_km',
          'lambda_zdw_nm', 'NF_I_DNN_dB', 'gain_I_DNN_dB',
          'NF_I_classical_dB', 'gain_I_classical_dB',
          'NF_II_DNN_dB', 'gain_II_DNN_dB',
          'NF_II_classical_dB', 'gain_II_classical_dB',
          'delta_gain_I_DNN_minus_classical_dB',
          'delta_NF_I_DNN_minus_classical_dB',
          'delta_gain_II_DNN_minus_classical_dB',
          'delta_NF_II_DNN_minus_classical_dB',
          'delta_gain_II_minus_I_DNN_dB', 'delta_NF_II_minus_I_DNN_dB',
          'delta_gain_II_minus_I_classical_dB', 'delta_NF_II_minus_I_classical_dB']


def sample_inputs(n=100, seed=42):
    """Independent uniforms from manuscript Table I; columns P, sep, beta2, ZDW."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError('n must be a positive integer')
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(19, 25, n), rng.uniform(4, 16, n),
                            rng.uniform(-10, -0.1, n), rng.uniform(1500, 1600, n)])


def compare_models(n=100, seed=42, output_dir=None, show=True):
    """Compute eight outputs per row and save CSV, summary, and two figures.

    Returns list of row dictionaries. Classical models use L=300 m and
    gamma=12 (W km)^-1. Checkpoints/preprocessing are loaded by predict.py.
    Errors are always DNN - classical within the same model. 
    Separate II - I columns compare models within each calculation method.
    A failed calculation raises with case/inputs; no point is silently dropped.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(output_dir) if output_dir is not None else BASE_DIR / 'results'
    out.mkdir(parents=True, exist_ok=True)
    inputs = sample_inputs(n, seed)
    rows = []
    print('All outputs/errors are in dB; errors = DNN - classical.', flush=True)
    print('case | I: NF_DNN NF_class gain_DNN gain_class dNF dGain | '
          'II: NF_DNN NF_class gain_DNN gain_class dNF dGain', flush=True)
    started = time.perf_counter()
    # Write each completed case immediately, retaining progress on interruption.
    with (out / 'comparison_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for k, (power, sep, beta2, zdw) in enumerate(inputs, start=1):
            row = dict(case=k, pump_power_dBm=float(power), pump_signal_sep_nm=float(sep),
                       beta2_ps2_per_km=float(beta2), lambda_zdw_nm=float(zdw))
            try:
                for model in ('I', 'II'):
                    nf_dnn, g_dnn = predict_nf_gain(sep, power, beta2, zdw, model=model)
                    nf_cl, g_cl = compute_nf_and_maxgain(power, sep, beta2, zdw, model,
                                                         fiber_length_m=300, gamma_Winv_km=12)
                    if not np.isfinite([nf_dnn, g_dnn, nf_cl, g_cl]).all():
                        raise RuntimeError(f'Nonfinite output from model {model}')
                    row.update({f'NF_{model}_DNN_dB': nf_dnn, f'gain_{model}_DNN_dB': g_dnn,
                                f'NF_{model}_classical_dB': nf_cl, f'gain_{model}_classical_dB': g_cl,
                                f'delta_gain_{model}_DNN_minus_classical_dB': g_dnn-g_cl,
                                f'delta_NF_{model}_DNN_minus_classical_dB': nf_dnn-nf_cl})
                for method in ('DNN', 'classical'):
                    for metric in ('gain', 'NF'):
                        row[f'delta_{metric}_II_minus_I_{method}_dB'] = (
                            row[f'{metric}_II_{method}_dB']-row[f'{metric}_I_{method}_dB'])
            except Exception as exc:
                raise RuntimeError(f'Case {k} failed: P={power}, sep={sep}, beta2={beta2}, ZDW={zdw}') from exc
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            pieces = []
            for model in ('I', 'II'):
                pieces.append(' '.join(f'{row[key]:9.5f}' for key in (
                    f'NF_{model}_DNN_dB', f'NF_{model}_classical_dB',
                    f'gain_{model}_DNN_dB', f'gain_{model}_classical_dB',
                    f'delta_NF_{model}_DNN_minus_classical_dB',
                    f'delta_gain_{model}_DNN_minus_classical_dB')))
            print(f'{k:4d} | '+ ' | '.join(pieces), flush=True)
    summary = {'n': n, 'seed': seed, 'error_definition': 'DNN minus classical (dB)',
               'fiber_length_m': 300, 'gamma_Winv_km': 12,
               'sampling': {'pump_power_dBm': [19, 25], 'pump_signal_sep_nm': [4, 16],
                            'beta2_ps2_per_km': [-10, -0.1], 'lambda_zdw_nm': [1500, 1600]},
               'metrics': {}}
    for metric in ('gain', 'NF'):
        fig, ax = plt.subplots(num=1 if metric == 'gain' else 2, figsize=(8, 5), clear=True)
        errors = {m: np.array([r[f'delta_{metric}_{m}_DNN_minus_classical_dB'] for r in rows])
                  for m in ('I', 'II')}
        # Shared edges allow a direct count comparison; retain every point.
        edges = np.histogram_bin_edges(np.concatenate(list(errors.values())), bins=20)
        for model, color in (('I', 'red'), ('II', 'blue')):
            ax.hist(errors[model], bins=edges, color=color, alpha=0.40,
                    edgecolor=color, label=f'Model {model} (N={n})')
            ax.hist(errors[model], bins=edges, color=color, histtype='step', linewidth=1.5)
            e = errors[model]
            stats = {'mean_error_dB': float(e.mean()), 'MAE_dB': float(np.abs(e).mean()),
                     'RMSE_dB': float(np.sqrt(np.mean(e**2))),
                     'max_abs_error_dB': float(np.abs(e).max())}
            summary['metrics'][f'{metric}_{model}'] = stats
            print(f'{metric}, model {model}: '+', '.join(f'{key}={v:.6g}' for key,v in stats.items()))
        name = 'maximum gain' if metric == 'gain' else 'noise figure'
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set(xlabel=f'{name.capitalize()} error: DNN - classical (dB)',
               ylabel='Number of cases', title=f'{name.capitalize()} prediction error')
        ax.legend()
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / ('fig1_gain_error.png' if metric == 'gain' else 'fig2_nf_error.png'), dpi=200)
    summary['elapsed_seconds'] = time.perf_counter()-started
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Completed {n} cases. Results saved to {out.resolve()}', flush=True)
    if show:
        plt.show()
    else:
        plt.close('all')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=BASE_DIR / 'results')
    parser.add_argument('--no-show', action='store_true', help='Save figures without opening windows')
    args = parser.parse_args()
    compare_models(args.n, args.seed, args.output_dir, show=not args.no_show)
