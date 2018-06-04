import argparse
from sklearn.manifold import TSNE
from sklearn import manifold
import matplotlib.pyplot as plt
from tsne import Tsne
from data_generator import read_from_file, read_from_db
import tkinter as tk


DEFAULT_DB_SRC = './datasets/networkdump-sorted.sqlite'


def apply_tsne(config):
    Y = TSNE(**config.get_config()).fit_transform(config.dataset)
    vis_x = Y[:, 0]
    vis_y = Y[:, 1]
    plt.scatter(vis_x, vis_y, c=config.target, s=10)
    plt.show()


def main(*args, **kwargs):
    if not args:
        print('Please provide a file name to read data from')
        return

    root = tk.Tk()

    def sel():
        selection = "Value = " + str(var.get())
        label.config(text=selection)
        root.quit()
        root.destroy()
        tsne = Tsne(manifold.TSNE(learning_rate=1000, init='random', perplexity=var.get()))
        tsne.plot(args[0], args[1], args[2])

    root.geometry('200x100')
    var = tk.DoubleVar()
    scale = tk.Scale(root, variable=var, orient="horizontal", from_=0, to=100)
    scale.pack(anchor=tk.CENTER)

    button = tk.Button(root, text="Set Perplexity", command=sel)
    button.pack(anchor=tk.CENTER)

    label = tk.Label(root)
    label.pack()
    root.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='app')
    parser.add_argument('--input')
    parser.add_argument('--colors')
    input_file = parser.parse_args().input
    color_file = parser.parse_args().colors
    if input_file is None:
        input_data, input_target = read_from_db(DEFAULT_DB_SRC)
        main(input_data, input_target, color_file)
    else:
        input_data, input_target = read_from_file(input_file)
        main(input_data, input_target, color_file)
