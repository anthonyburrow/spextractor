from spextractor import Spextractor


def test_prediction(file_optical, plot_dir, can_plot):
    params = {
        'z': 0.0459,
        'plot': True,
    }
    spex = Spextractor(file_optical, **params)

    spex.create_model(downsampling=3.)

    fig, ax = spex.plot

    name = 'test_prediction'
    ax.set_title(f'{name}')
    fn = f'{plot_dir}/{name}.png'
    if can_plot:
        fig.savefig(fn)


def test_process(file_optical, plot_dir, can_plot):
    params = {
        'z': 0.0459,
        'plot': True,
    }
    spex = Spextractor(file_optical, **params)

    spex.create_model(downsampling=3.)
    spex.process()

    fig, ax = spex.plot

    name = 'test_process'
    ax.set_title(f'{name}')
    fn = f'{plot_dir}/{name}.png'
    if can_plot:
        fig.savefig(fn)
