from sklearn import datasets
import numpy as np
import csv
import sqlite3


DB_NUM_ROWS = 10000


def create_db_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return None


def get_db_dataset(conn):
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
        data += cur.execute(f'''SELECT DISTINCT * FROM flow_features WHERE name = {curr_name} AND ''' +
                            f'''abs(CAST(random() AS REAL)) ''' +
                            f'''/ 9223372036854775808 < 0.5 LIMIT {DB_NUM_ROWS}''').fetchall()
    print("Loaded all the data")
    return data


def read_from_db(db_file):
    conn = create_db_connection(db_file)
    if conn is None:
        exit(1)
    db_rows = get_db_dataset(conn)

    data = []
    target = []
    for row in db_rows:
        vals = row[1:]
        tar = row[0]
        data.append(list(map(np.float64, vals)))
        target.append(tar)
    return np.copy(data), np.copy(target)


def read_from_file(filename):
    data = []
    target = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            vals = row[:-1]
            tar = row[-1]
            data.append(list(map(np.float64, vals)))
            target.append(tar)
    return np.array(data), np.array(target)


def main(*args, **kwargs):
    l = datasets.load_digits()
    digits = l.data
    targets = l.target
    for digit, target in list(map(lambda x: tuple(x), list(zip(digits, targets)))):
        row = ','.join(list(map(str, digit)))
        row += ',' + str(target)
        print(row)


if __name__ == '__main__':
    main()
