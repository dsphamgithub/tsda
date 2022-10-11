"""Testing script to compare the performance of datasets at different augmentation levels."""

import os
import argparse
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly
from tqdm import tqdm

from detection_experiment import damage_experiment, distance_experiment, sequence_experiment

current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--gt_file', default='/home/allenator/Pawsey-Internship/datasets/sgts_sequences/_single_annotations_array.npy', 
                    help='Ground truth annotations for dataset as a numpy file')
parser.add_argument('--eval_files', default='/home/allenator/Pawsey-Internship/eval_dir/sgts_sequences/augments.json',
                    help='Json of augment_level:file_path pairs, where file_path is for eval numpy files')
parser.add_argument('--num_frames', default=8, type=int, help='Number of frames per sequence the dataset')
parser.add_argument('--experiment', default='damage', choices=['damage', 'distance', 'sequence'] , help='Type of experiment to evaluate')
parser.add_argument('--metric', default='mAP', choices=['AP50','mAP', 'Mean IOU', 'Mean Score'] , help='Type of metric to evaluate')

if __name__ == "__main__":
    args = parser.parse_args()
    
    is_distance_experiment = args.experiment == 'distance'
    is_sequence_experiment = args.experiment == 'sequence'
    is_damage_experiment = args.experiment == 'damage'
    
    try:
        gt_array = np.array(np.load(args.gt_file), dtype=np.float32)
    except ValueError:
        # Flatten each entry into a single numpy array
        gt_array = np.array(np.load(args.gt_file, allow_pickle=True))
        gt_array = np.array([np.hstack([np.array(i) for i in x]) for x in gt_array], dtype=np.float32)
    
    with open(args.eval_files) as f:
        eval_json = json.load(f)
    
    augment_dict = {}
    
    # Get data frames for each augment level
    for aug in tqdm(eval_json):
        npy_path = eval_json[aug]
        pred_array = np.array(np.load(npy_path), dtype=np.float32)
        if args.experiment == 'damage':
            df = damage_experiment(gt_array, pred_array)
        elif args.experiment == 'distance':
            df = distance_experiment(gt_array, pred_array)
        elif args.experiment == 'sequence':
            df = sequence_experiment(gt_array, pred_array)
        augment_dict[aug] = df
    
    fig = px.scatter(title=args.experiment.capitalize() + ' vs ' + args.metric)

    # Axis labels   
    fig.update_yaxes(title='Mean Average Precision (mAP)', title_font=dict(size=16))
    if is_damage_experiment or is_sequence_experiment:
        fig.update_xaxes(title_text='Damage Ratio', title_font=dict(size=18))
    elif is_distance_experiment:
        fig.update_xaxes(title_text='Area of Sign in Pixels', title_font=dict(size=18))
        
    fig.update_layout(
        legend=dict(
            title='Proportion of synthetic images'),  
        font_size=14,
        font_family="Helvetica",
        font_color="black",
        title_font_size=20,
    )
    
    # Plot given experiment with new line for each augmentation level
    for aug in augment_dict:
        df = augment_dict[aug]
        if is_damage_experiment or is_sequence_experiment:
            fig = fig.add_trace(go.Scatter(x = df['Damage'], y = df[args.metric], 
                                        name=aug, mode='lines+markers'))
        elif is_distance_experiment:
            fig = fig.add_trace(go.Scatter(x = df['Area'], y = df[args.metric], 
                                        name=aug, mode='lines+markers'))
    plotly.io.write_image(fig, 'fig1.png', format='png', scale=2.5, width=1000, height=500)

    cwd = os.getcwd()
    name = args.gt_file.split('.')[0].split(os.sep)[-2]
    fig.write_html(f"{cwd}/{name}.html")
    with open(f"{cwd}/{name}.txt", 'w') as f:
        f.write(str(df))
