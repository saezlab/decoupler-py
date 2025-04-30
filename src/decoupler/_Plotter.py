import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure


class Plotter:
    def __init__(
        self,
        ax: Axes | None = None,
        figsize: tuple | None = (4, 3),
        dpi: int = 100,
        return_fig: bool = False,
        save: str | None = None,
    ) -> Figure | None:
        """
        Base class for plotters.

        Parameters
        ----------
        ax
            An existing :class:`matplotlib.axes._axes.Axes` instance to plot on. If ``None``,
            a new figure and axes will be created.
        figsize
            Size of the figure in inches as (width, height).
        dpi
            Dots per inch for the figure resolution.
        return_fig
            If ``True``, plotting methods should return the figure object instead of showing it.
        save
            If set, path to save the plot automatically to a file.

        Returns
        -------
        If ``return_fig=True``, returns :class:`matplotlib.figure.Figure` instance.
        """
        # Validate
        assert isinstance(ax, Axes) or ax is None, \
        'ax must be matplotlib.axes._axes.Axes or None'
        assert isinstance(figsize, tuple), \
        'figsize must be tuple'
        assert isinstance(dpi, (int, float)) and dpi > 0, \
        'dpi must be numerical and > 0'
        assert isinstance(return_fig, bool), \
        'return_fig must be bool'
        assert isinstance(save, str) or save is None, \
        'save must be str or None'
        self.ax = ax
        self.figsize = figsize
        self.dpi = dpi
        self.return_fig = return_fig
        self.save = save
        if self.ax is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi, tight_layout=True)
        else:
            self.fig = self.ax.figure

    def _return(self):
        if self.save is not None:
            self.fig.savefig(self.save, bbox_inches='tight')
        if self.return_fig:
            return self.fig
