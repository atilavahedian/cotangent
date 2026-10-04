"""Scientific figures from the complete V5 study, with every pair retained."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
r=json.loads((ROOT/'artifacts/v5/analysis/results.json').read_text())
assert r['runs']==200 and r['integrity_checks_passed'] and r['checkpoint_file_hashes_verified']
out=ROOT/'artifacts/v5/figures';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors={'native-fused':'#595e68','native-bucket':'#006c67'}

def save(fig,name):
    for ext in ('png','svg'):fig.savefig(out/f'{name}.{ext}')
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11.8,4.5),layout='constrained')
for ax,xkey,label,title in zip(axes,('elapsed_seconds','step'),
    ('Elapsed seconds incl. setup + probes','Optimizer step'),
    ('Same validation quality, earlier','Same training budget')):
    for arm in colors:
        points=[p for p in r['mean_learning_curves'] if p['arm']==arm]
        ax.plot([p[xkey] for p in points],[p['validation_bpb'] for p in points],
            color=colors[arm],label=points[0]['method'],linewidth=2)
    ax.axhline(3.2,color='#ac5c10',linestyle='--',linewidth=1,label='Frozen target: 3.2 BPB')
    ax.set(title=title,xlabel=label,ylabel='Validation BPB (lower is better)')
    ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.suptitle('Cotangent V5 · unchanged exact V4 method · 100 fresh paired seeds')
save(fig,'learning-curves')

fig,axes=plt.subplots(1,2,figsize=(11.8,4.1),layout='constrained')
p=r['small']
for ax,key,pairkey,scale,threshold,xlabel,title in (
    (axes[0],'elapsed_reduction_fraction','elapsed_reduction_fraction',100,10,
     'Paired time saved (%) · right is faster','Elapsed to the same 3.2-BPB target'),
    (axes[1],'quality_difference_bpb','test_difference_bpb',1,.01,
     'Candidate − native test BPB · left is better','Full reused-corpus test quality')):
    value=p[key]
    if value is not None:
        # Fixed seed-order jitter exposes all observations, without selection.
        ax.scatter([scale*pair[pairkey] for pair in p['pairs']],
            [((index%10)-4.5)/45 for index in range(100)],marker='|',s=100,color='#8eaea8',alpha=.7,label='All 100 paired observations')
        ax.errorbar(scale*value['mean'],.28,xerr=[[scale*(value['mean']-value['lower'])],[scale*(value['upper']-value['mean'])]],
            fmt='o',color='#006c67',capsize=5,label='Mean ± 99.5% paired t interval')
    else:ax.text(.5,.5,'Target censored',transform=ax.transAxes,ha='center')
    ax.axvline(0,color='#777',linewidth=.8)
    ax.axvline(threshold,color='#ac5c10',linestyle='--',linewidth=1,label='Unchanged decision threshold')
    ax.set(yticks=[],ylim=(-.25,.5),xlabel=xlabel,title=title);ax.grid(axis='x',alpha=.15);ax.legend(fontsize=8,loc='upper left')
fig.suptitle('V5 fixed-size replication · 99.5% two-sided intervals · no seed removal')
save(fig,'paired-outcomes')
print(str(out.relative_to(ROOT)))
