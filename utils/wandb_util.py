import os
import uuid

import wandb


def wandb_and_log_init(args):  # , group_name, run_name):

    run_folder = os.path.join(args.base_log_folder, "results", args.project_name, args.model)
    run_folder = os.path.join(run_folder, str(uuid.uuid4()).split('-')[0])
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    if args.wandb:
        comment = None if args.comment is None else ' '.join(args.comment)
        wandb.init(project=args.project_name, notes=comment, )
        wandb.config.update(args)

    return run_folder