import sklearn
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.animation import FuncAnimation
from helpers import *


class Tsne(object):
    """
    This is a custom tSNE object that can be used to animate tSNE as it progresses through its
    iterations. Furthermore, it can also be used to zoom in, which invokes another process where
    this new tSNE is run on the zoomed-in points only. In order to see the progression of the
    algorithm, the default gradient descent method had to be replaced with a custom one so that
    plot points could be accumulated.
    """
    def __init__(self, tsne, should_animate=True):
        self.tsne = tsne
        self.is_fit = False
        self.should_animate = should_animate

    def get_steps(self, X, y):
        """
        :param X: The input data to which tSNE is applied
        :param y: The corresponding input targets for tSNE
        :return: A list which contains the the various positions of tSNE as it
                 ran through its iterations
        based on https://github.com/oreillymedia/t-SNE-tutorial
        """
        old_grad = sklearn.manifold.t_sne._gradient_descent
        positions = []

        # Replace old gradient func
        sklearn.manifold.t_sne._gradient_descent = create_custom_gradient_descent(positions)
        _ = self.tsne.fit_transform(X)
        self.is_fit = True

        # return old gradient descent back
        sklearn.manifold.t_sne._gradient_descent = old_grad
        return positions

    def plot(self, X, y, color_map_file=None):
        print("Fitting t-SNE...")
        out = self.tsne.fit_transform(X)
        print("t-SNE fit finished!")
        self.is_fit = True

        y_mapping = {i: n for i, n in enumerate(set(y))}

        limits = np.max(out, axis=0), np.min(out, axis=0)

        fig = plt.figure()
        fig.clf()
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111)
        ax.set_xlim([limits[1][0], limits[0][0]])
        ax.set_ylim([limits[1][1], limits[0][1]])
        ax.callbacks.connect('xlim_changed', create_x_limits_func(out, X, y, ax))
        ax.callbacks.connect('ylim_changed', create_y_limits_func(out, X, y, ax))

        jet = plt.get_cmap('jet')
        c_norm = colors.Normalize(vmin=0, vmax=len(y_mapping))
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)
        A, B = np.array(list(zip(*out.reshape(-1, 2))))

        dots_list = []

        color_mapping = {} if color_map_file is None else get_color_mapping(color_map_file)

        for i, idx in enumerate(sorted(list(y_mapping.keys()))):
            val = y_mapping[idx]
            if color_map_file is None:
                color_val = scalar_map.to_rgba(i)
            else:
                color_val = color_mapping[val]
            a, b = A[y == val], B[y == val]
            dots, = ax.plot(b, a, 'o', color=color_val, label=val)
            color_mapping[val] = color_val
            dots_list.append(dots)
            plt.legend(loc='best')

        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)

        write_color_mapping(color_mapping)
        plt.show()

    def animate(self, X, y, use_tqdm=1, filename=None, color_map_file=None):
        pos = self.get_steps(X, y)
        y_mapping = {i: n for i, n in enumerate(set(y))}
        last_iter = pos[-1].reshape(-1, 2)
        limits = np.max(last_iter, axis=0), np.min(last_iter, axis=0)

        fig = plt.figure()
        fig.clf()
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111)

        jet = plt.get_cmap('jet')
        c_norm = colors.Normalize(vmin=0, vmax=len(y_mapping))
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)
        A, B = np.array(list(zip(*pos[0].reshape(-1, 2))))
        dots_list = []

        color_mapping = {} if color_map_file is None else get_color_mapping(color_map_file)
        for i, (idx, val) in enumerate(y_mapping.items()):
            color_val = scalar_map.to_rgba(i)
            a, b = A[y == val], B[y == val]
            dots, = ax.plot(b, a, 'o', color=color_val, label=val)
            dots_list.append(dots)
            plt.legend(loc='best')

        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)

        write_color_mapping(color_mapping)

        def init():
            ax.set_xlim([limits[1][0], limits[0][0]])
            ax.set_ylim([limits[1][1], limits[0][1]])
            ax.callbacks.connect('xlim_changed', create_x_limits_func(last_iter, X, y, ax))
            ax.callbacks.connect('ylim_changed', create_y_limits_func(last_iter, X, y, ax))
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
        elif use_tqdm == 1:
            from tqdm import tqdm
            frames = tqdm(np.arange(0, len(pos)-1))
        elif use_tqdm == 2:
            from tqdm import tqdm_notebook
            frames = tqdm_notebook(np.arange(0, len(pos)-1))

        animation = FuncAnimation(fig, update, frames=frames, init_func=init,
                                  interval=50, repeat=False)
        if filename is None:
            plt.show()
        else:
            animation.save(filename, dpi=80, writer='imagemagick')
