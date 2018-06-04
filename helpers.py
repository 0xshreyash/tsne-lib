from sklearn import datasets
import numpy as np
from numpy import linalg
import sklearn
from sklearn import manifold
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.animation import FuncAnimation
import csv
import subprocess

class TsneConfigBuilder(object):
    def __init__(self):
        self.config = {'n_components': 2,
                       'verbose': 1,
                       'n_iter': 100,
                       'perplexity': 50
                      }
        digits = datasets.load_digits()
        self.dataset = digits.data
        self.target = digits.target

    def set_n_components(self, n):
        self.config['n_components'] = n

    def set_verbose(self, v):
        self.config['verbose'] = v

    def set_n_iter(self, i):
        self.config['n_iter'] = i

    def set_perplexity(self, p):
        self.config['perplexity'] = p

    def set_dataset(self, d):
        self.dataset = d

    def set_target(self, t):
        self.target = t

    def get_config(self):
        return self.config


class Tsne(object):
    def __init__(self, tsne, should_animate=True):
        self.tsne = tsne
        self.isfit = False
        self.should_animate = should_animate

    def get_steps(self, X, y):
        # based on https://github.com/oreillymedia/t-SNE-tutorial
        old_grad = sklearn.manifold.t_sne._gradient_descent
        positions = []

        def _gradient_descent(objective, p0, it, n_iter, objective_error=None,
                              n_iter_check=1, n_iter_without_progress=50,
                              momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                              min_grad_norm=1e-7, min_error_diff=1e-7,
                              verbose=0, args=None, kwargs=None):
            if args is None:
                args = []
            if kwargs is None:
                kwargs = {}

            p = p0.copy().ravel()
            update = np.zeros_like(p)
            gains = np.ones_like(p)
            error = np.finfo(np.float).max
            best_error = np.finfo(np.float).max
            best_iter = 0

            for i in range(it, n_iter):
                # We save the current position.
                positions.append(p.copy())

                new_error, grad = objective(p, *args, **kwargs)
                grad_norm = linalg.norm(grad)

                inc = update * grad >= 0.0
                dec = np.invert(inc)
                gains[inc] += 0.05
                gains[dec] *= 0.95
                np.clip(gains, min_gain, np.inf)
                grad *= gains
                update = momentum * update - learning_rate * grad
                p += update

                if (i + 1) % n_iter_check == 0:
                    if new_error is None:
                        new_error = objective_error(p, *args)
                    error_diff = np.abs(new_error - error)
                    error = new_error

                    if verbose >= 2:
                        m = "[t-SNE] Iteration %d: error = %.7f, gradient norm = %.7f"
                        print(m % (i + 1, error, grad_norm))

                    if error < best_error:
                        best_error = error
                        best_iter = i
                    elif i - best_iter > n_iter_without_progress:
                        if verbose >= 2:
                            print("[t-SNE] Iteration %d: did not make any progress "
                                  "during the last %d episodes. Finished."
                                  % (i + 1, n_iter_without_progress))
                        break
                    if grad_norm <= min_grad_norm:
                        if verbose >= 2:
                            print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                                  % (i + 1, grad_norm))
                        break
                    if error_diff <= min_error_diff:
                        if verbose >= 2:
                            m = "[t-SNE] Iteration %d: error difference %f. Finished."
                            print(m % (i + 1, error_diff))
                        break

                if new_error is not None:
                    error = new_error

            return p, error, i

        # Replace old gradient func
        sklearn.manifold.t_sne._gradient_descent = _gradient_descent
        _ = self.tsne.fit_transform(X)
        self.isfit = True

        # return old gradient descent back
        sklearn.manifold.t_sne._gradient_descent = old_grad
        return positions

    def plot(self, X, y):
        print("Starting T-SNE fit")
        out = self.tsne.fit_transform(X)
        print("T-SNE fit finished!")
        self.isfit = True
        y_mapping = {i: n for i, n in enumerate(set(y))}

        lims = np.max(out, axis=0), np.min(out, axis=0)
        print("lims:", lims)

        def on_xlims_change(_):
            x_right = ax.get_xlim()[1]
            x_left = ax.get_xlim()[0]
            y_bottom = ax.get_ylim()[0]
            y_top = ax.get_ylim()[1]

            def in_bounds(pt):
                return (pt[0] <= x_right and pt[0] >= x_left and
                        pt[1] >= y_bottom and pt[1] <= y_top)

            indxs = []
            for i, pt in enumerate(out):
                if in_bounds(pt):
                    indxs.append(i)

            new_points = []
            new_ys = []
            for i in indxs:
                new_points.append(X[i])
                new_ys.append(y[i])


        def on_ylims_change(_):
            x_right = ax.get_xlim()[1]
            x_left = ax.get_xlim()[0]
            y_bottom = ax.get_ylim()[0]
            y_top = ax.get_ylim()[1]

            def in_bounds(pt):
                return (pt[0] <= x_right and pt[0] >= x_left and
                        pt[1] >= y_bottom and pt[1] <= y_top)

            indxs = []
            for i, pt in enumerate(out):
                if in_bounds(pt):
                    indxs.append(i)

            new_points = []
            new_ys = []
            for i in indxs:
                new_points.append(np.ndarray.tolist(X[i]))
                new_ys.append(y[i])

            # print("got new ys:", new_ys)
            # print("got these many new points:", len(new_points))
            with open('output.csv', 'w') as csvfile:
                csvfile.seek(0)
                csvfile.truncate()
                writer = csv.writer(csvfile, delimiter=',')

                for row in list(map(lambda x: x[0] + [x[1]],
                                    list(zip(new_points, new_ys)))):
                    writer.writerow(list(map(str, row)))

            subprocess.run(['python3', 'app.py', '--input', 'output.csv'])

        fig = plt.figure()
        fig.clf()
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111)
        ax.set_xlim([lims[1][0], lims[0][0]])
        ax.set_ylim([lims[1][1], lims[0][1]])
        ax.callbacks.connect('xlim_changed', on_xlims_change)
        ax.callbacks.connect('ylim_changed', on_ylims_change)

        jet = plt.get_cmap('jet')
        c_norm = colors.Normalize(vmin=0, vmax=len(y_mapping))
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)
        A, B = np.array(list(zip(*out.reshape(-1, 2))))

        dots_list = []

        print("y_mapping:")
        print(y_mapping)
        print("---------------------------------")

        count_dict = {}
        for i, (idx, val) in enumerate(y_mapping.items()):
            color_val = scalar_map.to_rgba(i)
            a, b = A[y == val], B[y == val]
            dots, = ax.plot(b, a, 'o', color=color_val, label=val)
            dots_list.append(dots)
            count_dict[val] = len(b) if val not in count_dict else count_dict[val]+len(b)
            plt.legend(loc=1)
        print("count dict is:", count_dict)

        plt.show()

    def animate(self, X, y, use_tqdm=1, filename=None):
        pos = self.get_steps(X, y)
        print("Got total points (len(X)):", len(X))

        y_mapping = {i: n for i, n in enumerate(set(y))}
        # y_mapping = sorted(list(set(y)))
        last_iter = pos[-1].reshape(-1, 2)
        # print("last iter:", last_iter)
        lims = np.max(last_iter, axis=0), np.min(last_iter, axis=0)
        print("lims:", lims)

        fig = plt.figure()
        fig.clf()
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111)

        jet = plt.get_cmap('jet')
        c_norm = colors.Normalize(vmin=0, vmax=len(y_mapping))
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)
        A, B = np.array(list(zip(*pos[0].reshape(-1, 2))))
        # A, B = np.array(list(zip(*pos[-1].reshape(-1, 2))))
        dots_list = []

        print("y_mapping:")
        print(y_mapping)
        print("---------------------------------")

        # count_dict = {}
        # for i, (key, _) in enumerate(y_mapping.items()):
        for i, (idx, val) in enumerate(y_mapping.items()):
            color_val = scalar_map.to_rgba(i)
            # a, b = A[y == i], B[y == i]
            a, b = A[y == val], B[y == val]
            # count_dict[val] = len(b) if val not in count_dict else count_dict[val]+len(b)
            # print("got a, b", a, b)
            dots, = ax.plot(b, a, 'o', color=color_val, label=val)
            dots_list.append(dots)
            plt.legend(loc=1)

        # print("count dict is:", count_dict)

        def on_xlims_change(_):
            x_right = ax.get_xlim()[1]
            x_left = ax.get_xlim()[0]
            y_bottom = ax.get_ylim()[0]
            y_top = ax.get_ylim()[1]

            def in_bounds(pt):
                return (pt[0] <= x_right and pt[0] >= x_left and
                        pt[1] >= y_bottom and pt[1] <= y_top)

            indxs = []
            for i, pt in enumerate(last_iter):
                if in_bounds(pt):
                    indxs.append(i)

            new_points = []
            new_ys = []
            for i in indxs:
                new_points.append(X[i])
                new_ys.append(y[i])


        def on_ylims_change(_):
            x_right = ax.get_xlim()[1]
            x_left = ax.get_xlim()[0]
            y_bottom = ax.get_ylim()[0]
            y_top = ax.get_ylim()[1]

            def in_bounds(pt):
                return (pt[0] <= x_right and pt[0] >= x_left and
                        pt[1] >= y_bottom and pt[1] <= y_top)

            indxs = []
            for i, pt in enumerate(last_iter):
                if in_bounds(pt):
                    indxs.append(i)

            new_points = []
            new_ys = []
            for i in indxs:
                new_points.append(np.ndarray.tolist(X[i]))
                new_ys.append(y[i])

            # print("got new ys:", new_ys)
            # print("got these many new points:", len(new_points))
            with open('output.csv', 'w') as csvfile:
                csvfile.seek(0)
                csvfile.truncate()
                writer = csv.writer(csvfile, delimiter=',')

                for row in list(map(lambda x: x[0] + [x[1]], \
                                    list(zip(new_points, new_ys)))):
                    writer.writerow(list(map(str, row)))

            subprocess.run(['python', 'app.py', '--input', 'output.csv'])


        def init():
            ax.set_xlim([lims[1][0], lims[0][0]])
            ax.set_ylim([lims[1][1], lims[0][1]])
            ax.callbacks.connect('xlim_changed', on_xlims_change)
            ax.callbacks.connect('ylim_changed', on_ylims_change)
            # print('The length of the list to be plotted is: ', len(dots_list))
            return [i for i in dots_list]

        def update(i):
            for j in range(len(dots_list)):
                aa, bb = np.array(list(zip(*pos[i].reshape(-1, 2))))
                aa, bb = aa[y == y_mapping[j]], bb[y == y_mapping[j]]
                dots_list[j].set_xdata(aa)
                dots_list[j].set_ydata(bb)
            return [i for i in dots_list]+[ax]

        if use_tqdm == 0:
            frames = np.arange(0, len(pos)-1)
            # frames = np.arange(0, 1)
        elif use_tqdm == 1:
            from tqdm import tqdm
            frames = tqdm(np.arange(0, len(pos)-1))
            # frames = tqdm(np.arange(0, 1))
        elif use_tqdm == 2:
            from tqdm import tqdm_notebook
            frames = tqdm_notebook(np.arange(0, len(pos)-1))
            # frames = tqdm_notebook(np.arange(0, 1))
        anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                             interval=50, repeat=False)
        if filename is None:
            plt.show()
        else:
            anim.save(filename, dpi=80, writer='imagemagick')
