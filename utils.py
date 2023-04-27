'''
@file: util.py

This file contains all the utility functions required for the BTP.

@author: Rukmangadh Sai Myana
@mail: rukman.sai@gmail.com
'''

import os
import torch


def set_defaults(given_args, default_args):
    '''
    Set default values to the arguments that are None.

    This function is necessary because same named arguments (from command
    prompt flags) can have different defaults in different contexts.

    @param given_args: The parameters given by the flags.
    @param default_args: The default values for the args.
    '''
    for key, value in given_args.items():
        if value is None:
            given_args[key] = default_args[key]
    return given_args


def get_summary_dir(args):
    '''
    Return the directory for the summary writer to store the tensorboard
    summary.

    @param args: The arguments passed as flags from the command prompt.
    @returns: The path to the directory where the summary is gonna be stored
    for the current experiment.
    '''
    summary_dir = os.path.join(args.logdir, args.dataset,
                               args.task + str(args.task_number))

    # directory doesn't exist
    if not os.path.isdir(os.path.join(summary_dir, 'experiment_1')):
        summary_dir = os.path.join(summary_dir, 'experiment_1')
        os.makedirs(summary_dir)
    # directory exists
    else:
        prev_experiment_dirname = max(os.listdir(summary_dir))
        if args.resume:
            # resuming training means no new logging directory
            new_experiment_dirname = prev_experiment_dirname
        else:
            new_experiment_dirname = prev_experiment_dirname[:-1] + \
                str(int(prev_experiment_dirname[-1]) + 1)
        summary_dir = os.path.join(summary_dir, new_experiment_dirname)
    return summary_dir


def save_best_model(save_dir, model, metrics, use_metric='accuracy'):
    '''
    Save the model in the given directory if it's better than the older model
    in the directory.

    @param save_dir: The directory to save the model in.
    @param model: The model to save
    @param metrics: The metrics for the model.
    @param use_metric: The metric to be used for comparing the models.
    Default is 'accuracy'
    '''
    filepath = os.path.join(save_dir, 'best_model.pt')

    # older best model file exists
    if os.path.exists(filepath):
        best_model = torch.load(filepath)
        best_metric = best_model['metrics'][use_metric]
        if best_metric < metrics[use_metric]:
            # save file contains the state dict and metrics also
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
            }, filepath)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
        }, filepath)