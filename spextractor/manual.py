import matplotlib.pyplot as plt


class ManualRange:

    def __init__(self, w, f, default_lines, logger):
        self.w = w
        self.f = f
        self.def_lines = default_lines
        self.logger = logger

        self.cid = None

        self.fig, self.ax = plt.subplots()

        self.selected_feature = None
        self.n_selections = 0
        self.new_bounds = []

        self.init_plot()
        plt.show()

    def init_plot(self):
        self.ax.plot(self.w, self.f)
        self.fig.set_size_inches(12, 7)
        self.ax.set_title('Select feature to change', fontsize=16)

        for feature in self.def_lines:
            self.ax.axvline(self.def_lines[feature]['rest'], ls='--')

        self.cid = self.fig.canvas.mpl_connect('button_press_event',
                                               self.select_feature)

    def select_feature(self, event):
        dists = ((abs(event.xdata - self.def_lines[feat]['rest']), feat)
                 for feat in self.def_lines)
        closest_feature = min(dists)[1]
        print('Selected %s' % closest_feature)

        self.fig.canvas.mpl_disconnect(self.cid)
        self.init_bound_change(closest_feature)

    def init_bound_change(self, feature):
        self.selected_feature = feature

        self.ax.set_title('Select lower bound 1', fontsize=16)
        self.fig.canvas.draw()

        self.cid = self.fig.canvas.mpl_connect(
            'button_press_event', self.select_bound)

    def select_bound(self, event):
        lam = event.xdata
        print('Chosen %.3f' % lam)
        self.new_bounds.append(lam)

        c = '#ff7e21'
        if self.n_selections == 0:
            self.ax.set_title('Select lower bound 2', fontsize=16)
        if self.n_selections == 1:
            self.ax.set_title('Select higher bound 1', fontsize=16)
        if self.n_selections == 2:
            self.ax.set_title('Select higher bound 2', fontsize=16)

        self.ax.axvline(lam, c=c)

        self.fig.canvas.draw()
        self.n_selections += 1

        if self.n_selections == 4:
            self.end_bound_change()

    def end_bound_change(self):
        self.fig.canvas.mpl_disconnect(self.cid)

        new_info = {}
        new_info['rest'] = self.def_lines[self.selected_feature]['rest']
        new_info['lo_range'] = tuple(self.new_bounds[0:2])
        new_info['hi_range'] = tuple(self.new_bounds[2:4])

        self.def_lines[self.selected_feature] = new_info
        self.logger.info('%s bounds changed to %s' % (self.selected_feature,
                                                      str(new_info)))

        self.selected_feature = None
        self.n_selections = 0
        self.new_bounds = []

        self.ax.clear()
        self.init_plot()
