
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('ticks')
cmap = sns.color_palette()

if __name__ == '__main__':
    """
    Plot model evolution
    """

    # add argument parser
    parser = argparse.ArgumentParser(description='Show evaluation plot.')
    parser.add_argument('results', metavar='N', type=str, nargs='+', help='result.pkl files.')
    parser.add_argument('--acc', help='evaluate accuracy.', action='store_true')
    parser.add_argument('--perc', help='show percentage value.', action='store_true')
    parser.add_argument('--max_epoch', type=int, default=None, help='last epoch to plot.')
    parser.add_argument('--ymin', help='minimum y value.', type=float, default=None)
    parser.add_argument('--ymax', help='maximum y value.', type=float, default=None)
    parser.add_argument('--watch', help='refresh plot.', action='store_true')
    parser.add_argument('--plot_all', help='plot loss and accuracy.', action='store_true')
    args = parser.parse_args()

    while True:

        # load results
        all_results = dict()
        for result in args.results:
            exp_name = result.split(os.sep)[-1].split('.pkl')[0]
            with open(result, 'rb') as fp:
                try:
                    exp_res = pickle.load(fp)
                except(EOFError):
                    break
                all_results[exp_name] = exp_res

        # Accuracy plot, disable loss plot
        if not args.acc:
            plot_loss = True
        else:
            plot_loss = False

        # present results
        plt.figure("Model Evolution")
        plt.clf()
        # setup subplots for accuracy + loss
        if args.plot_all:
            ax = plt.subplot(212)

            ax2 = plt.subplot(211)
            args.acc = True
            plot_loss = True
        else:
            ax = plt.subplot(111)

        plt.subplots_adjust(bottom=0.15, left=0.15, right=0.9, top=0.95)

        # python 2 - 3 compatibility workaround
        try:
            ar_iter = all_results.iteritems()
        except AttributeError:
            ar_iter = all_results.items()

        for i, (exp_name, exp_res) in enumerate(ar_iter):

            # Accuracy plot
            if args.acc:

                if args.max_epoch is not None:
                    max_epoch = int(args.max_epoch)
                    exp_res['tr_accs'] = exp_res['tr_accs'][0:max_epoch]
                    exp_res['va_accs'] = exp_res['va_accs'][0:max_epoch]

                # train accuracy
                tr_accs = np.asarray(exp_res['tr_accs'], dtype=np.float)
                tr_accs[np.equal(tr_accs, None)] = np.nan
                indices = np.nonzero(~np.isnan(tr_accs))[0]
                tr_accs = tr_accs[indices]
                if args.perc:
                    acc = " (%.2f%%)" % tr_accs[-1]
                    label = exp_name + '_tr' + acc
                else:
                    label = exp_name + '_tr'

                if args.plot_all:
                    ax2.plot(indices, tr_accs, '-', color=cmap[i], linewidth=3, alpha=0.8, label=label)
                else:
                    ax.plot(indices, tr_accs, '-', color=cmap[i], linewidth=3, alpha=0.8, label=label)

                # validation accuracy
                va_accs = np.asarray(exp_res['va_accs'], dtype=np.float)
                va_accs[np.equal(va_accs, None)] = np.nan
                indices = np.nonzero(~np.isnan(va_accs))[0]
                va_accs = va_accs[indices]
                if args.perc:
                    acc = " (%.2f%%)" % np.mean(va_accs[-10::])
                    label = exp_name + '_va' + acc
                else:
                    label = exp_name + '_va'

                if args.plot_all:
                    ax2.plot(indices, va_accs, '-', color=cmap[i], linewidth=2, label=label)
                else:
                    ax.plot(indices, va_accs, '-', color=cmap[i], linewidth=2, label=label)

                # plot maximum accuracy
                max_acc = np.max(va_accs)
                if args.plot_all:
                    ax2.plot([indices[0], indices[-1]], [max_acc] * 2, '--', color=cmap[i], alpha=0.5)
                    ax2.text(indices[-1], max_acc, ('%.2f' % max_acc), va='bottom', ha='right', color=cmap[i])
                else:
                    ax.plot([indices[0], indices[-1]], [max_acc] * 2, '--', color=cmap[i], alpha=0.5)
                    ax.text(indices[-1], max_acc, ('%.2f' % max_acc), va='bottom', ha='right', color=cmap[i])

            # loss plot
            if plot_loss:
                ax.plot(exp_res['pred_tr_err'], '-', color=cmap[i], linewidth=3, alpha=0.8, label=exp_name + '_tr')
                ax.plot(exp_res['pred_val_err'], '-', color=cmap[i], linewidth=2, label=exp_name + '_va')

                # plot minimum validation loss
                min_loss = np.min(exp_res['pred_val_err'])
                ax.plot([0, len(exp_res['pred_val_err']) - 1], [min_loss] * 2, '--', color=cmap[i], alpha=0.5)
                ax.text(len(exp_res['pred_val_err']) - 1, min_loss, ('%.5f' % min_loss), va='top', ha='right', color=cmap[i])
                ax.plot(np.argmin(exp_res['pred_val_err']), min_loss, 'o', color=cmap[i])

        if args.acc:
            if args.plot_all:
                ax2.set_ylabel("Accuracy", fontsize=20)
                ax2.legend(loc="lower right", fontsize=18).draggable()
                ax2.set_ylim([args.ymin, 102])
            else:
                ax.set_ylabel("Accuracy", fontsize=20)
                ax.legend(loc="upper left", fontsize=18).draggable()
                ax.set_ylim([args.ymin, 102])

        if plot_loss:
            ax.set_ylabel("Loss", fontsize=20)
            ax.legend(loc="upper right", fontsize=20).draggable()

        if args.ymin is not None and args.ymax is not None:
            ax.set_ylim([args.ymin, args.ymax])

        if args.max_epoch is not None:
            ax.set_xlim([0, args.max_epoch])

        ax.set_xlabel("Epoch", fontsize=20)
        ax.grid('on')

        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        if args.plot_all:
            ax2.grid('on')
            ax2.tick_params(axis='x', labelsize=18)
            ax2.tick_params(axis='y', labelsize=18)

        plt.draw()

        if args.watch:
            plt.pause(10.0)
        else:
            ax.show(block=True)
            break
