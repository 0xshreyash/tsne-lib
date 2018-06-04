import subprocess
import csv
import numpy as np
from numpy import linalg


COLOR_MAP_FILE = './.color-map.conf'


def create_in_bounds_func(ax):
    def func(pt):
        x_right = ax.get_xlim()[1]
        x_left = ax.get_xlim()[0]
        y_bottom = ax.get_ylim()[0]
        y_top = ax.get_ylim()[1]
        return (x_right >= pt[0] >= x_left and
                y_bottom <= pt[1] <= y_top)
    return func


def create_x_limits_func(last_iter, X, y, ax):
    in_bounds = create_in_bounds_func(ax)

    def func(_):
        indexes = []
        for i, pt in enumerate(last_iter):
            if in_bounds(pt):
                indexes.append(i)
        new_points = []
        new_ys = []
        for i in indexes:
            new_points.append(X[i])
            new_ys.append(y[i])
    return func


def create_y_limits_func(last_iter, X, y, ax):
    in_bounds = create_in_bounds_func(ax)

    def func(_):
        indexes = []
        for i, pt in enumerate(last_iter):
            if in_bounds(pt):
                indexes.append(i)
        new_points = []
        new_ys = []
        for i in indexes:
            new_points.append(np.ndarray.tolist(X[i]))
            new_ys.append(y[i])

        with open('output.csv', 'w') as csv_file:
            csv_file.seek(0)
            csv_file.truncate()
            writer = csv.writer(csv_file, delimiter=',')

            for row in list(map(lambda x: x[0] + [x[1]],
                                list(zip(new_points, new_ys)))):
                writer.writerow(list(map(str, row)))

        subprocess.run(['python3', 'app.py', '--input', 'output.csv', '--colors', COLOR_MAP_FILE])
    return func


def get_color_mapping(color_map_file):
    with open(color_map_file, 'r') as f:
        d = {}
        for row in f:
            row = row.split(',')
            d[row[0]] = tuple(map(float, row[1:]))
        return d


def write_color_mapping(color_mapping):
    with open(COLOR_MAP_FILE, 'w') as f:
        for k, v in color_mapping.items():
            f.write(','.join([str(k)] + [str(val) for val in v]) + '\n')


def create_custom_gradient_descent(positions):
    def func(objective, p0, it, n_iter, objective_error=None,
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

    return func
