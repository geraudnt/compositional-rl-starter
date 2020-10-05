import argparse
import time
import datetime
import torch
import tensorboardX
import sys

import utils
from model import ACModel
from torch_ac.algos import a2c, ppo


if __name__ == '__main__':
    # Parse arguments

    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--env", default='BabyAI-GoToObjCustom-v0',
                        help="name of the environment to train on (default: BabyAI-GoToObjCustom-v0)")
    parser.add_argument("--obj_type", type=str, default=None,
                        help="object type")
    parser.add_argument("--obj_color", type=str, default=None,
                        help="object color")
    parser.add_argument("--algo", default='ppo',
                        help="algorithm to use: a2c | ppo (default: ppo)")                    
    parser.add_argument("--load", default=False, action='store_true',
                        help="load the model (default: false)")                    
    parser.add_argument("--model", default='model',
                        help="name of the model (default: model)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")

    ## Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")

    args = parser.parse_args()

    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    model_name = args.model
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.obj_type, args.obj_color, seed=args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        if not args.load:
            status = {"num_frames": 0, "update": 0}
        else:
            status = utils.get_status(model_dir)
            txt_logger.info("Training status loaded\n")
    except OSError:
        status = {"num_frames": 0, "update": 0}

    # Load observations preprocessor

    obs_space, preprocess_obs_goals = utils.get_obs_goals_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obs_goals.vocab.load_vocab(status["vocab"])
    txt_logger.info("observations preprocessor loaded")

    # Load model

    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
        txt_logger.info("Model loaded\n")
    acmodel.to(device)
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == "a2c":
        algo = a2c.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obs_goals)
    elif args.algo == "ppo":
        algo = ppo.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obs_goals)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")
    if "agent_goals" in status:
        algo.goals = status["agent_goals"]
    txt_logger.info("Goals loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            success_per_episode = utils.synthesize([1 if r > 0 else 0 for r in logs["return_per_episode"]])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration", "goals"]
            data = [update, num_frames, fps, duration, len(list(algo.goals.keys()))]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "success_rate", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], success_per_episode["mean"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | G {} | rR:uomM {:.2f} {:.2f} {:.2f} {:.2f} | F:uomM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | S {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data).encode('utf-8'))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update, "model_state": acmodel.state_dict(),
                    "agent_goals": algo.goals, "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obs_goals, "vocab"):
                status["vocab"] = preprocess_obs_goals.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
