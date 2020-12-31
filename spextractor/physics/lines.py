sn_lines = {
    'Ca II H&K': {'rest': 3945.12,
                  'lo_range': (3450, 3800),
                  'hi_range': (3800, 3950)},
    'Si 4000A': {'rest': 4129.73,
                 'lo_range': (3840, 3950),
                 'hi_range': (4000, 4200)},
    'Mg II 4300A': {'rest': 4481.2,
                    'lo_range': (4000, 4250),
                    'hi_range': (4300, 4700)},
    'Fe II 4800A': {'rest': 5083.42,
                    'lo_range': (4300, 4700),
                    'hi_range': (4950, 5600)},
    'S W': {'rest': 5536.24,
            'lo_range': (5050, 5300),
            'hi_range': (5500, 5750)},
    'Si II 5800A': {'rest': 6007.7,
                    'lo_range': (5400, 5700),
                    'hi_range': (5800, 6000)},
    'Si II 6150A': {'rest': 6355.1,
                    'lo_range': (5800, 6100),
                    'hi_range': (6200, 6600)},
    'Fe II': {'rest': 5169,
              'lo_range': (4950, 5050),
              'hi_range': (5150, 5250)},
    'He I': {'rest': 5875,
             'lo_range': (5350, 5450),
             'hi_range': (5850, 6000)},
    'O I': {'rest': 7773,
            'lo_range': (7250, 7350),
            'hi_range': (7750, 7950)}
}

sn_types = {
    'Ia': ('Ca II H&K', 'Si 4000A', 'Mg II 4300A', 'Fe II 4800A',
           'S W', 'Si II 5800A', 'Si II 6150A'),
    'Ib': ('Fe II', 'He I'),
    'Ic': ('Fe II', 'O I')
}


def get_features(sn_type: str):
    lines = sn_types[sn_type]

    line_info = {}
    for line in lines:
        line_info[line] = sn_lines[line]

    return line_info
