sn_lines = {
    'Ca II H&K': {'rest': 3945.12,
                  'lo_range': (3500., 3750.),
                  'hi_range': (3800., 4000.)},
    'Si II 4000A': {'rest': 4129.73,
                 'lo_range': (3800., 4000.),
                 'hi_range': (4050., 4250.)},
    'Mg II 4300A': {'rest': 4481.2,
                    'lo_range': (4250., 4300.),
                    'hi_range': (4300., 4420.)},
    # 'Si III 4450A': {'rest': 4481.2,
    #                 'lo_range': (4300., 4420.),
    #                 'hi_range': (4500., 4600.)},
    'Fe II 4800A': {'rest': 5083.42,
                    'lo_range': (4300., 4700.),
                    'hi_range': (4950., 5600.)},
    'S II 5500A': {'rest': 5536.24,
            'lo_range': (5200., 5300.),
            'hi_range': (5500., 5650.)},
    'Si II 5800A': {'rest': 6007.7,
                    'lo_range': (5600., 5750.),
                    'hi_range': (5800., 6000.)},
    'Si II 6150A': {'rest': 6355.1,
                    'lo_range': (5800., 6100.),
                    'hi_range': (6250., 6500.)},
    'O I 7500A': {'rest': 7773.,
            'lo_range': (7250., 7350.),
            'hi_range': (7500., 7750.)},
    'Fe II': {'rest': 5169.,
              'lo_range': (4950., 5050.),
              'hi_range': (5150., 5250.)},
    'He I': {'rest': 5875.,
             'lo_range': (5350., 5450.),
             'hi_range': (5850., 6000.)}
}

sn_types = {
    'Ia' : (
        'Ca II H&K',
        'Si II 4000A',
        'Mg II 4300A',
        # 'Si III 4450A',
        'Fe II 4800A',
        'S II 5500A',
        'Si II 5800A',
        'Si II 6150A',
        'O I 7500A'
    ),
    'Ib' : (
        'Fe II',
        'He I'
    ),
    'Ic' : (
        'Fe II',
        'O I 7500A'
    )
}


def get_features(sn_type: str):
    lines = sn_types[sn_type]

    line_info = {}
    for line in lines:
        line_info[line] = sn_lines[line]

    return line_info
