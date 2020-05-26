import argparse
import os
import time
import sys
import copy

import warnings
warnings.simplefilter("ignore", UserWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import ffjord.lib.utils as utils
import ffjord.lib.layers.odefunc as odefunc

from ffjord.train_misc import standard_normal_logprob
from ffjord.train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from ffjord.train_misc import add_spectral_norm, spectral_norm_power_iteration
from ffjord.train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from ffjord.train_misc import build_model_tabular

from gu import *
from util import *

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='pinwheel'
)
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--niters', type=int, default=50000, help='Total iteration numbers in training.')
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size in training.')
parser.add_argument('--eval_size', type=int, default=100000, help='Sample size in evaluation.')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--beta1', type=float, default=0.9, help='Beta 1 in Adam.')
parser.add_argument('--beta2', type=float, default=0.999, help='Beta 2 in Adam.')

parser.add_argument('--clr', action='store_true', help='Use cyclic LR in training.')
parser.add_argument('--clr_size_up', type=int, default=2000, help='Size of up step in cyclic LR.')
parser.add_argument('--clr_scale', type=int, default=3, help='Scale of base lr in cyclic LR.')

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

# parser.add_argument('--save', type=str, default=result_path)
# parser.add_argument('--viz_freq', type=int, default=100)
# parser.add_argument('--val_freq', type=int, default=100)
# parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--log_interval', type=int, default=1000, help='How often to show loss statistics and save models/samples.')

parser.add_argument('--gu_num', type=int, default=8, help='Components of GU clusters.')


def get_transforms(model):

    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn


def compute_loss(args, model, dataloader, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size

    x = dataloader.get_sample(batch_size)
    zero = torch.zeros(x.shape[0], 1).to(x)

    # transform to z
    z, delta_logp = model(x, zero)

    # compute log q(z)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss


if __name__ == '__main__':
    args = parser.parse_args()

    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(rootPath)

    search_type = 'manual'
    experiment = f'gu{args.gu_num}/ffjord/{args.niters}'

    model_path = os.path.join(rootPath, search_type, 'models', experiment)
    image_path = os.path.join(rootPath, search_type, 'images', experiment)
    makedirs(model_path, image_path)
    log_path = model_path + '/logs'
    logger = get_logger(log_path)

    if args.layer_type == "blend":
        logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
        args.time_length = 1.0

    logger.info(args)

    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    if args.gu_num == 8:
        dataloader = GausUniffMixture(n_mixture=args.gu_num, mean_dist=10, sigma=2, unif_intsect=1.5, unif_ratio=1.,
                                      device=args.device, extend_dim=False)
    else:
        dataloader = GausUniffMixture(n_mixture=args.gu_num, mean_dist=5, sigma=0.1, unif_intsect=5, unif_ratio=3,
                                      device=args.device, extend_dim=False)

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, 1, regularization_fns).to(args.device)
    if args.spectral_norm: add_spectral_norm(model)
    set_cnf_options(args, model)

    # logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                 betas=(args.beta1, args.beta2))
    if args.clr:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr / args.clr_scale, max_lr=args.lr,
                                                      step_size_up=args.clr_size_up, cycle_momentum=False)
    else:
        scheduler = None

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    nfef_meter = utils.RunningAverageMeter(0.93)
    nfeb_meter = utils.RunningAverageMeter(0.93)
    tt_meter = utils.RunningAverageMeter(0.93)

    start = time.time()
    best_loss = float('inf')
    model.train()

    for i in range(1, args.niters + 1):
        optimizer.zero_grad()
        if args.spectral_norm: spectral_norm_power_iteration(model, 1)

        loss = compute_loss(args, model, dataloader, args.batch_size)
        loss_meter.update(loss.item())

        if len(regularization_coeffs) > 0:
            reg_states = get_regularization(model, regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
            )
            loss = loss + reg_loss

        total_time = count_total_time(model)
        nfe_forward = count_nfe(model)

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        nfe_total = count_nfe(model)
        nfe_backward = nfe_total - nfe_forward
        nfef_meter.update(nfe_forward)
        nfeb_meter.update(nfe_backward)

        time_meter.update(time.time() - start)
        tt_meter.update(total_time)

        log_message = (
            'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | NFE Forward {:.0f}({:.1f})'
            ' | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
                i, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, nfef_meter.val, nfef_meter.avg,
                nfeb_meter.val, nfeb_meter.avg, tt_meter.val, tt_meter.avg
            )
        )
        if len(regularization_coeffs) > 0:
            log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)

        logger.info(log_message)

        if i % args.log_interval == 0:
            with torch.no_grad():
                real = dataloader.get_sample(args.eval_size)

                sample_fn, density_fn = get_transforms(model)
                z = torch.randn(args.eval_size, 1).type(torch.float32).to(args.device)
                zk, inds = [], torch.arange(0, z.shape[0]).to(torch.int64)
                for ii in torch.split(inds, int(10000)):
                    zk.append(sample_fn(z[ii]))
                fake = torch.cat(zk, 0)

                w_distance_real = w_distance(real, fake)

                # eval_loss = compute_loss(args, model, dataloader, batch_size=args.eval_size)
                # eval_nfe = count_nfe(model)
                #
                # logger.info(f'Iter {i} / {args.niters}, Time {round(time.time() - start, 4)},  '
                #             f'w_distance_real: {w_distance_real}, test loss: {round(eval_loss.item(), 5)}, '
                #             f'test NFE: {round(eval_nfe, 1)}')

                cur_state_path = os.path.join(model_path, str(i))
                torch.save(model, cur_state_path + '_' + 'ffjord.pth')

                real_sample = real.cpu().data.numpy().squeeze()
                fake_sample = fake.cpu().data.numpy().squeeze()

                # plot.
                plt.cla()
                fig = plt.figure(figsize=(FIG_W, FIG_H))
                ax = fig.add_subplot(111)
                ax.set_facecolor('whitesmoke')
                ax.grid(True, color='white', linewidth=2)

                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.get_xaxis().tick_bottom()

                # _sample = np.concatenate([real_sample, fake_sample])
                kde_num = 200
                min_real, max_real = min(real_sample), max(real_sample)
                kde_width_real = kde_num * (max_real - min_real) / args.eval_size
                min_fake, max_fake = min(fake_sample), max(fake_sample)
                kde_width_fake = kde_num * (max_fake - min_fake) / args.eval_size
                sns.kdeplot(real_sample, bw=kde_width_real, label='Data', color='green', shade=True, linewidth=6)
                sns.kdeplot(fake_sample, bw=kde_width_fake, label='Model', color='orange', shade=True, linewidth=6)

                ax.set_title(f'True EM Distance: {w_distance_real}.', fontsize=FONTSIZE)
                ax.legend(loc=2, fontsize=FONTSIZE)
                ax.set_ylabel('Estimated Density by KDE', fontsize=FONTSIZE)
                ax.tick_params(axis='x', labelsize=FONTSIZE * 0.7)
                ax.tick_params(axis='y', labelsize=FONTSIZE * 0.5, direction='in')

                cur_img_path = os.path.join(image_path, str(i) + '.jpg')
                plt.tight_layout()

                plt.savefig(cur_img_path)
                plt.close()

                # plt.cla()
                # fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))
                # fig.subplots_adjust(top=0.80)
                #
                # ax = fig.add_subplot(111)
                # _sample = np.concatenate([real_sample, fake_sample])
                # x_min, x_max = min(_sample), max(_sample)
                # range_width = x_max - x_min
                # kde_num = 200
                # kde_width = kde_num * range_width / args.eval_size
                # sns.kdeplot(real_sample, bw=kde_width, label='Estimated Density by KDE: Real', color='skyblue',
                #             shade=True)
                # sns.kdeplot(fake_sample, bw=kde_width, label='Estimated Density by KDE: Fake', color='red', shade=True)
                # ax.set_title(f'W_distance_real: {w_distance_real}', fontsize=FONTSIZE)
                # ax.legend(loc=2, fontsize=FONTSIZE)
                # ax.set_ylabel('Estimated Density by KDE', fontsize=FONTSIZE)
                # ax.tick_params(labelsize=FONTSIZE)
                #
                # cur_img_path = os.path.join(image_path, str(i) + '.jpg')
                # plt.savefig(cur_img_path)
                # plt.close()

        start = time.time()

    logger.info('Finish All...')
