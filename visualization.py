from collections import defaultdict
from typing import Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

model_list = [
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en"
]
sns.set_theme(font_scale=1.35)
sns.color_palette("rocket")

def load_data(cpu_type: str) -> pd.DataFrame:
    model_dict = {x: model_list for x in ['python', 'cpp', 'q5_0']}
    param_count = {'tiny.en': 39, 'base.en': 74, 'small.en': 244, 'medium.en': 768}
    avgs = []
    for arch_type, models in model_dict.items():
        for model in models:
            try:
                result_df = pd.read_csv(f"data/{cpu_type}/{arch_type}_{model}.csv")
            except FileNotFoundError:
                continue
            avg_wer = result_df['WER'].mean()
            avg_runtime = result_df['runtime'].mean()
            avgs.append({'Model Type': arch_type, 'parameters': param_count[model], 'model': model, 'avg_runtime': avg_runtime, 'avg_wer': avg_wer})
    return pd.DataFrame.from_records(avgs)


def make_acc_graphs(cpu_type: str, df: pd.DataFrame, ax_lim: Union[Tuple[float, float], None] = None):
    ax = sns.lineplot(data=df, y='avg_wer', x='parameters', hue='Model Type', marker='o')
    ax.set(xlabel='Parameters (M)', ylabel='Word Error Rate', title=f'Whisper Transcription Accuracy: ({cpu_type})')
    if ax_lim:
        ax.set_ylim(ax_lim)
    plt.tight_layout()
    plt.savefig(f'{cpu_type}_accuracy_results.png')
    plt.close()
    return ax.get_ylim()
        
def make_runtime_graphs(cpu_type: str, df: pd.DataFrame, ax_lim: Union[Tuple[float, float], None] = None):
    ax = sns.lineplot(data=df, y='avg_wer', x='avg_runtime', hue='Model Type', marker='o')
    ax.set(xlabel='Runtime (s)', ylabel='Word Error Rate', title=f'Whisper Runtime Tradeoff: ({cpu_type})')
    if ax_lim:
        ax.set_ylim(ax_lim)
    plt.tight_layout()
    plt.savefig(f'{cpu_type}_runtime_results.png')
    plt.close()

def main() -> None:
    ylims = []
    for cpu in ['x86', 'arm']:
        df = load_data(cpu)
        # We want to capture the first y limit and use it for all subsequent graphs
        if ylims:
            ylims.append(make_acc_graphs(cpu, df, ylims[0]))
            make_runtime_graphs(cpu, df)
        else:
            ylims.append(make_acc_graphs(cpu, df))
            make_runtime_graphs(cpu, df)

    

if __name__ == '__main__':
    main()
