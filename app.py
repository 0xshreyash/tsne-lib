import argparse
import sqlite3
from sklearn.manifold import TSNE
from sklearn import manifold, datasets
import numpy as np
import matplotlib.pyplot as plt
from helpers import Tsne
import csv
import tkinter as tk


DB_NUM_ROWS = 10000


def create_db_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return None


def get_db_dataset(conn, limit):
    print("Getting Data")
    cur = conn.cursor()
    data = []

    devices = ["\"SwannOne SoundView Outdoor Camera\"",
               "\"DLink Camera\"",
               "\"Google-Home\"",
               "\"WeMo Switch\"",
               "\"SwannOne Smart Hub\"",
               "\"Philips Hue Bulbs\"",
               "\"rpi-bonesi\""
               ]
    for curr_name in devices:
        print("Extracting row from database:", curr_name)
        data += cur.execute(f'''SELECT DISTINCT * FROM flow_features WHERE name = {curr_name} AND abs(CAST(random() AS REAL)) 
                                / 9223372036854775808 < 0.5 LIMIT {limit}''').fetchall()
    print("Loaded all the data")
    return data


def apply_tsne(config):
    Y = TSNE(**config.get_config()).fit_transform(config.dataset)
    vis_x = Y[:, 0]
    vis_y = Y[:, 1]
    plt.scatter(vis_x, vis_y, c=config.target, s=10)
    plt.show()


def read_db_data(db_rows):
    data = []
    target = []
    for row in db_rows:
        vals = row[1:]
        tar = row[0]
        data.append(list(map(np.float64, vals)))
        target.append(tar)
    return np.copy(data), np.copy(target)


def read_csv_data(filename):
    data = []
    target = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            vals = row[:-1]
            tar = row[-1]
            data.append(list(map(np.float64, vals)))
            target.append(tar)
    return np.copy(data), np.copy(target)


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
        tsne.plot(args[0], args[1])

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
    input_file = parser.parse_args().input
    if input_file is None:
        connection = create_db_connection('datasets/networkdump-sorted.sqlite')
        if connection is None:
            exit(1)
        db_output = get_db_dataset(connection, DB_NUM_ROWS)
        data, target = read_db_data(db_output)
        main(data, target)
    else:
        data, target = read_csv_data(input_file)
        # print("got target:", all(target==datasets.load_digits().target))
        # print("got data:", all(data.ravel()==datasets.load_digits().data.ravel()))
        main(data, target)
