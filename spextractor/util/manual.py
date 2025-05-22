import matplotlib.pyplot as plt


class ManualRange:

    def __init__(self, spectrum, feature_list, logger):
        self.w = spectrum.wave
        self.f = spectrum.flux
        self.feature_list = feature_list
        self.logger = logger

        self.cid = None

        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=150)

        self.selected_feature = None
        self.n_selections = 0
        self.new_bounds = []

        self.init_select_feature()

        plt.tight_layout()

        self.fig.show()
        while plt.fignum_exists(self.fig.number):
            plt.pause(0.1)

    def init_plot(self):
        self.ax.plot(self.w, self.f, 'k-')

        spacing = (self.w[-1] - self.w[0]) * 0.05
        wave_left = self.w[0] - spacing
        wave_right = self.w[-1] + spacing
        self.ax.set_xlim(wave_left, wave_right)
        self.ax.set_ylim(0., 1.05)

    def init_select_feature(self):
        self.init_plot()

        self.ax.set_title('Select feature to change')

        y_min, y_max = self.ax.get_ylim()
        y_pos = y_min + 0.4 * (self.f.min() - y_min)
        for feature in self.feature_list:
            x_left = feature.wave_left
            x_right = feature.wave_right

            self.ax.axvline(x_left, ls='-', color='tab:blue', alpha=0.3)
            self.ax.axvline(x_right, ls='-', color='tab:blue', alpha=0.3)
            self.ax.axvspan(x_left, x_right, alpha=0.2, color='tab:blue')

            self.ax.text(x_left + 30., y_pos, feature.name, rotation=90.)

        self.fig.canvas.draw()

        self.cid = self.fig.canvas.mpl_connect(
            'button_press_event', self.select_feature
        )

    def select_feature(self, event):
        for feature in self.feature_list:
            if feature.wave_left < event.xdata < feature.wave_right:
                break
        else:
            print('Feature not found (click within feature bounds)')
            return

        print(f'Selected {feature.name}')

        self.fig.canvas.mpl_disconnect(self.cid)
        self.init_bound_change(feature)

    def init_bound_change(self, feature):
        self.selected_feature = feature

        self.ax.clear()
        self.init_plot()

        self.ax.set_title('Select lower bound 1')

        self.fig.canvas.draw()

        self.cid = self.fig.canvas.mpl_connect(
            'button_press_event', self.select_bound
        )

    def select_bound(self, event):
        lam = event.xdata
        print(f'Chosen {lam:.3f}')
        self.new_bounds.append(lam)

        if self.n_selections == 0:
            self.ax.set_title('Select lower bound 2')
        elif self.n_selections == 1:
            self.ax.set_title('Select higher bound 1')
        elif self.n_selections == 2:
            self.ax.set_title('Select higher bound 2')

        self.ax.axvline(lam, c='#ff7e21')

        self.fig.canvas.draw()
        self.n_selections += 1

        if self.n_selections == 4:
            self.end_bound_change()

    def end_bound_change(self):
        self.fig.canvas.mpl_disconnect(self.cid)

        new_lo_range = tuple(self.new_bounds[0:2])
        new_hi_range = tuple(self.new_bounds[2:4])
        self.selected_feature.update_endpoints(new_lo_range, new_hi_range)

        msg = (
            f'{self.selected_feature} bounds changed to:\n'
            f'{new_lo_range}, {new_hi_range}'
        )
        self.logger.info(msg)

        self.selected_feature = None
        self.n_selections = 0
        self.new_bounds = []

        self.ax.clear()
        self.init_select_feature()
