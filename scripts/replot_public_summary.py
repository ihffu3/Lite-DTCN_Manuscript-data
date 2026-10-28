"""Replot released clean-history means; no CSI access or neural inference."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
STYLE = {
    'ar': ('AR', '#666666', '^'),
    'kalman': ('AR-Kalman (clean)', '#aaaaaa', 'v'),
    'lmmse': ('PC-Wiener', '#222222', 'D'),
    'esprit': ('Temporal ESPRIT', '#555555', 'P'),
    'lite': ('Lite-DTCN', '#D55E00', '*'),
    'stemgnn': ('STEM-GNN', '#009E73', 's'),
    'itransformer': ('iTransformer', '#CC79A7', 'D'),
    'wireless_linformer': ('Wireless LinFormer', '#56B4E9', 'P'),
}


def draw():
    with (ROOT/'data/summary/nmse_se_300_snr10_rzf.csv').open(encoding='utf-8',newline='') as f:
        rows = list(csv.DictReader(f))
    with (ROOT/'data/summary/facts.csv').open(encoding='utf-8',newline='') as f:
        facts = {r['fact_id']: float(r['value']) for r in csv.DictReader(f)}
    assert len(rows) == 16 and {(float(r['horizon_ms']),r['method']) for r in rows} == {
        (h,m) for h in (2.,4.) for m in STYLE
    }, 'Retain eight methods at both horizons.'
    with plt.rc_context({'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes = plt.subplots(1,2,figsize=(7.16,3.05),sharey=True)
        for ax,h in zip(axes,(2.,4.)):
            for r in rows:
                if float(r['horizon_ms']) != h: continue
                label,color,marker = STYLE[r['method']]
                ax.scatter(float(r['nmse_gain_db']),float(r['sum_se_gain']),
                           color=color,marker=marker,s=36,edgecolor='white',linewidth=.5)
            ax.axhline(0,color='#444444',linewidth=.6)
            # Zero prediction has 0-dB NMSE; its rate is not inferred.
            ax.axvline(facts[f'nmse_outdated_v300_h{h}'],color='#777777',ls='--',lw=.7)
            ax.set(xlim=(0,4.05),ylim=(-.52,.57),xticks=[0,1,2,3,4],
                   xlabel='NMSE gain over outdated CSI (dB)',title=f'300 km/h, {h:g} ms, 10 dB RZF')
            ax.grid(alpha=.2)
        axes[0].set_ylabel('Sum-SE gain over outdated CSI (bit/s/Hz)')
        handles=[Line2D([],[],color=c,marker=m,ls='',label=l) for l,c,m in STYLE.values()]
        handles.append(Line2D([],[],color='#777777',ls='--',label='Zero-prediction NMSE'))
        fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=7,
                   bbox_to_anchor=(.5,-.015),columnspacing=1.15)
        fig.subplots_adjust(left=.09,right=.995,top=.90,bottom=.26,wspace=.28)
        out=ROOT/'figures';out.mkdir(exist_ok=True)
        for suffix in ('pdf','svg'):
            fig.savefig(out/f'nmse_se_tradeoff.{suffix}',bbox_inches='tight',pad_inches=.025)
        plt.close(fig)
        print('Wrote figures/nmse_se_tradeoff.pdf and .svg')


if __name__ == '__main__':
    draw()
