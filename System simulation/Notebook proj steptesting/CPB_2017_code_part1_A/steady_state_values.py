import csv


def steady_state():
    ss_values = {}

    with open('steady_states.csv', 'r') as csvfile:
        ss = csv.reader(csvfile, delimiter=',')

        for row in ss:
            ss_values[row[0]] = float(row[1])

    return ss_values
