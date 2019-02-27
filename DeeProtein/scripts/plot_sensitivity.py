import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import pandas as pd
import json
import os
import sys
from Bio import AlignIO 
import random



style_folder = "/code/DeeProtein/style/"
with open(style_folder + "style_seq_plotter.json", "r") as style_file, open(style_folder + "colors.json", "r") as color_file:
    plt.style.use(json.load(style_file))
    colors = json.load(color_file)
    color_list = [*colors.values()]



def plot_annotated_sequence_track(sequence=None,
                                  idx_range=None,
                                  data_dict=None,
                                  left_label="Sensitivity",
                                  title="",
                                  use_name=False,
                                  left_y_lim=None,
                                  right_y_lim=None,
                                  std_y_lim=None,
                                  xoffset=0,
                                  seqoffset=0,
                                  width=7.06,
                                  height=1.4,
                                  show_seq=True,
                                  show_pos=True, 
                                  lw=2
                                  ):
    if idx_range is None:
        start_index = 1
        end_index = len(sequence)
    else:
        assert idx_range[0] < idx_range[1]
        start_index = idx_range[0]
        end_index = idx_range[1]
        
    width_per_aa = 0.08
    base_width = 0.5
    
    if not width:
        width = base_width + width_per_aa*(end_index-start_index) 
    
    fig = plt.figure(figsize=(width, height), dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)
        
    x_idx = np.arange(start_index, end_index + 1)
    x_slice = slice(start_index-1, end_index)
    mean_in_window = np.nanmean(np.asarray([entry["data"][x_slice] for entry in data_dict.values()]), axis=0)


    assert isinstance(data_dict, dict)
    for plot_idx, (key, entry) in enumerate(data_dict.items()):
        assert "data" in entry
        if use_name:
            assert "name" in entry
        color = entry.get("color", None)
        if color:
            color = colors[color]
        else:
            color = color_list[plot_idx]

        ax = ax1
            
        label = key if not 'label' in entry else entry['label']
        
        if entry.get("bar", False):
            ax.bar(x=x_idx, height=entry["data"][x_slice], color=color, width=0.6, label=label) 
        else:
            ax.plot(x_idx, entry["data"][x_slice], color=color, lw=lw, label=label)
    
    # zero line
    ax1.plot([min(x_idx), max(x_idx)], [0.0, 0.0], color="black", label="Baseline", lw=1)
    if left_y_lim:
        ax1.set_ylim(left_y_lim)

    if show_seq:
        seq_letters = [' ']*seqoffset + [l for l in sequence[x_slice]]
    
        # TWO TICK LABELS LEVELS: (with sequence:)
        text_ticks = plt.FixedFormatter(seq_letters)
        ax1.xaxis.set_minor_locator(plt.FixedLocator(x_idx))
        x_idx_major = [x for x in x_idx if (x-x_idx[0])%10 == 0]
        if show_pos:
            x_idx_major_labels = [x + xoffset for x in x_idx_major]
            
        else:
            x_idx_major_labels = [None for x in x_idx_major]
        major_ticks = plt.FixedFormatter(x_idx_major_labels)
        ax1.xaxis.set_minor_formatter(text_ticks)
        ax1.xaxis.set_major_locator(plt.FixedLocator(x_idx_major))
        ax1.xaxis.set_major_formatter(major_ticks)
    else:
    # ONLY POSITION:
        x_idx_minor = [1] + [x for x in x_idx if (x-(x_idx[0]-1))%100 == 0]
        x_idx_minor_labels = [x + xoffset for x in x_idx_minor]
        ax1.set_xticks(x_idx_minor)
        ax1.set_xticklabels(x_idx_minor_labels )

    ax1.set_ylabel(left_label)    
    ax1.set_title(title)
    
    if show_seq:
        ax1.tick_params(axis="x", which="major", length=0, pad=15)
    # resize
    fig.set_dpi(300)
    fig.set_size_inches(width, height, forward=True)  
    
    
    # legend figure:
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(bbox_to_anchor=(0.015, 1.01, 1., .101),
               columnspacing=1, 
               loc='upper left', 
               ncol=len(handles),
              )
    
    
    ax1.set_xlim(1, end_index)


    return fig, mean_in_window


if __name__ == '__main__':
    sen_data = sys.argv[1]
    gos = sys.argv[2].split(',')
    path_out = sys.argv[3]
    
    if len(sys.argv) > 4:
        xoffset = int(sys.argv[4])
    else:
        xoffset = 0

    data = pd.read_table(sen_data, sep='\t')
    sequence = data.loc[:, 'AA'].iloc[1:]
    
    data_dict = {}
    for go in gos:
        data_dict[go] = {"data": data[go].iloc[1:]}

    fig, _ = plot_annotated_sequence_track(sequence=sequence, 
                                           data_dict=data_dict, 
                                           idx_range=None, 
                                           xoffset=xoffset,
                                           width=None)
    print('Writing plot to {}'.format(path_out))
    plt.savefig(path_out)