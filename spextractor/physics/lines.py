FEATURE_REST_WAVES: dict[str, float] = {
    'Ca II H&K': 3945.12,
    'Si II 4000A': 4129.73,
    'Mg II 4300A': 4481.2,
    # 'Si III 4450A': 4481.2,
    'Fe II 4800A': 5083.42,
    'S II 5500A': 5536.24,
    'Si II 5800A': 5972.9449706,
    'Si II 6150A': 6355.1,
    'O I 7500A': 7773.0,
    'Fe II': 5169.0,
    'He I': 5875.0,
}

FEATURE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    'Ca II H&K': {
        'lo_range': (3500.0, 3750.0),
        'hi_range': (3800.0, 4000.0),
    },
    'Si II 4000A': {
        'lo_range': (3800.0, 4000.0),
        'hi_range': (4050.0, 4250.0),
    },
    'Mg II 4300A': {
        'lo_range': (4250.0, 4300.0),
        'hi_range': (4300.0, 4420.0),
    },
    'Fe II 4800A': {
        'lo_range': (4300.0, 4700.0),
        'hi_range': (4950.0, 5600.0),
    },
    'S II 5500A': {
        'lo_range': (5200.0, 5300.0),
        'hi_range': (5500.0, 5650.0),
    },
    'Si II 5800A': {
        'lo_range': (5500.0, 5800.0),
        'hi_range': (5850.0, 5980.0),
    },
    'Si II 6150A': {
        'lo_range': (5800.0, 6100.0),
        'hi_range': (6250.0, 6500.0),
        'lo_range_hv': (5700.0, 6000.0),
        'hi_range_hv': (6150.0, 6400.0),
    },
    'O I 7500A': {
        'lo_range': (7250.0, 7350.0),
        'hi_range': (7500.0, 7750.0),
    },
    'Fe II': {
        'lo_range': (4950.0, 5050.0),
        'hi_range': (5150.0, 5250.0),
    },
    'He I': {
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
) -> tuple[dict[str, float], dict[str, dict[str, tuple[float, float]]]]:
    """Return rest wavelengths and range dictionaries for a given SN type.

    Parameters
    ----------
    sn_type : str
        Supernova type key in `sn_types`.

    Returns
    -------
    (rest_map, ranges_map)
        rest_map: feature -> rest wavelength
        ranges_map: feature -> dict of range tuples (lo/hi plus optional hv
        ranges).
    """
    lines = sn_types[sn_type]
    rest = {name: FEATURE_REST_WAVES[name] for name in lines}
    ranges = {name: FEATURE_RANGES[name] for name in lines}
    return rest, ranges
