sn_lines: dict[str, dict[str, float | tuple[float, float]]] = {
    'Ca II H&K': {
        'rest': 3945.12,
        'lo_range': (3500.0, 3750.0),
        'hi_range': (3800.0, 4000.0),
    },
    'Si II 4000A': {
        'rest': 4129.73,
        'lo_range': (3800.0, 4000.0),
        'hi_range': (4050.0, 4250.0),
    },
    'Mg II 4300A': {
        'rest': 4481.2,
        'lo_range': (4250.0, 4300.0),
        'hi_range': (4300.0, 4420.0),
    },
    # 'Si III 4450A': {'rest': 4481.2,
    #                 'lo_range': (4300., 4420.),
    #                 'hi_range': (4500., 4600.)},
    'Fe II 4800A': {
        'rest': 5083.42,
        'lo_range': (4300.0, 4700.0),
        'hi_range': (4950.0, 5600.0),
    },
    'S II 5500A': {
        'rest': 5536.24,
        'lo_range': (5200.0, 5300.0),
        'hi_range': (5500.0, 5650.0),
    },
    'Si II 5800A': {
        'rest': 5972.9449706,
        'lo_range': (5500.0, 5800.0),
        'hi_range': (5850.0, 5980.0),
    },
    'Si II 6150A': {
        'rest': 6355.1,
        'lo_range': (5800.0, 6100.0),
        'hi_range': (6250.0, 6500.0),
        'lo_range_hv': (5700.0, 6000.0),
        'hi_range_hv': (6150.0, 6400.0),
    },
    'O I 7500A': {
        'rest': 7773.0,
        'lo_range': (7250.0, 7350.0),
        'hi_range': (7500.0, 7750.0),
    },
    'Fe II': {
        'rest': 5169.0,
        'lo_range': (4950.0, 5050.0),
        'hi_range': (5150.0, 5250.0),
    },
    'He I': {
        'rest': 5875.0,
        'lo_range': (5350.0, 5450.0),
        'hi_range': (5850.0, 6000.0),
    },
}

sn_types: dict[str, tuple[str, ...]] = {
    'Ia': (
        'Ca II H&K',
        'Si II 4000A',
        'Mg II 4300A',
        # 'Si III 4450A',
        'Fe II 4800A',
        'S II 5500A',
        'Si II 5800A',
        'Si II 6150A',
        'O I 7500A',
    ),
    'Ib': ('Fe II', 'He I'),
    'Ic': ('Fe II', 'O I 7500A'),
}


def get_features(
    sn_type: str,
) -> dict[str, dict[str, float | tuple[float, float]]]:
    lines = sn_types[sn_type]

    line_info = {}
    for line in lines:
        line_info[line] = sn_lines[line]

    return line_info
