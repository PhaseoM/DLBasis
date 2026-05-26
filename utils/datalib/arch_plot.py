import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from enum import Enum

# ======================================================
#  ArchPlot Blueprint
# ======================================================
#
# ArchPlot: List[Arch]
#
# Arch: BasicInfo,List[Tuple[GraphType, GraphData]]
# |- BasicInfo:
# |  |- title: str
# |  |- xlabel: str
# |  |- ylabel: str
# |  |-...
# |
# |- GraphType: Enum[str]
# |- GraphData: Tuple[GraphInfo, GraphEntity]
#   |- GraphInfo: list[ArchInfo]
#   |  |- arch_info: list[ArchInfo]     # [N, C]
#   |     |- name: str
#   |     |- color: str | None
#   |     |- style: str | None
#   |     |- ...
#   |
#   |- GraphEntity: list[ArchEntity]    # [N, M]
#
# ======================================================


class ArchInfo:
    def __init__(
        self,
        name: str | None = None,
        color: str | None = None,
        style: str | None = None,
        **kwargs,
    ):
        self.name = name
        self.color = color
        self.style = style
        self.kwargs = dict(kwargs)

    def set_name(self, name):
        self.name = name

    def set_color(self, color):
        self.color = color

    def set_style(self, style):
        self.style = style

    def set_kwargs(self, **kwargs):
        self.kwargs.update(kwargs)

    def to_kwargs(self):
        plot_kwargs = dict(self.kwargs)
        if self.name is not None:
            plot_kwargs.setdefault("label", self.name)
        if self.color is not None:
            plot_kwargs.setdefault("color", self.color)
        return plot_kwargs


class CurveInfo(ArchInfo):
    def __init__(
        self,
        name: str | None = None,
        color: str | None = None,
        style: str | None = None,
        linewidth: float | None = None,
        marker: str | None = None,
        markersize: float | None = None,
        alpha: float | None = None,
        **kwargs,
    ):
        super().__init__(name=name, color=color, style=style, **kwargs)
        self.linewidth = linewidth
        self.marker = marker
        self.markersize = markersize
        self.alpha = alpha

    def set_linewidth(self, linewidth):
        self.linewidth = linewidth

    def set_marker(self, marker):
        self.marker = marker

    def set_markersize(self, markersize):
        self.markersize = markersize

    def set_alpha(self, alpha):
        self.alpha = alpha

    def to_kwargs(self):
        plot_kwargs = super().to_kwargs()
        if self.style is not None:
            plot_kwargs.setdefault("linestyle", self.style)
        if self.linewidth is not None:
            plot_kwargs.setdefault("linewidth", self.linewidth)
        if self.marker is not None:
            plot_kwargs.setdefault("marker", self.marker)
        if self.markersize is not None:
            plot_kwargs.setdefault("markersize", self.markersize)
        if self.alpha is not None:
            plot_kwargs.setdefault("alpha", self.alpha)
        return plot_kwargs


class MarkerInfo(ArchInfo):
    def __init__(
        self,
        name: str | None = None,
        color: str | None = None,
        style: str | None = None,
        size: float | None = None,
        marker: str | None = None,
        alpha: float | None = None,
        edgecolors: str | None = None,
        linewidths: float | None = None,
        **kwargs,
    ):
        super().__init__(name=name, color=color, style=style, **kwargs)
        self.size = size
        self.marker = marker
        self.alpha = alpha
        self.edgecolors = edgecolors
        self.linewidths = linewidths

    def set_size(self, size):
        self.size = size

    def set_marker(self, marker):
        self.marker = marker

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_edgecolors(self, edgecolors):
        self.edgecolors = edgecolors

    def set_linewidths(self, linewidths):
        self.linewidths = linewidths

    def to_kwargs(self):
        plot_kwargs = super().to_kwargs()
        marker_style = self.marker if self.marker is not None else self.style
        if marker_style is not None:
            plot_kwargs.setdefault("marker", marker_style)
        if self.size is not None:
            plot_kwargs.setdefault("s", self.size)
        if self.alpha is not None:
            plot_kwargs.setdefault("alpha", self.alpha)
        if self.edgecolors is not None:
            plot_kwargs.setdefault("edgecolors", self.edgecolors)
        if self.linewidths is not None:
            plot_kwargs.setdefault("linewidths", self.linewidths)
        return plot_kwargs


class ArchEntity:
    def __init__(
        self,
        **kwargs,
    ):
        self.kwargs = dict(kwargs)

    def set_kwargs(self, **kwargs):
        self.kwargs.update(kwargs)

    def to_kwargs(self):
        return dict(self.kwargs)


class CurveEntity(ArchEntity):
    def __init__(
        self,
        x: list[float] | None = None,
        y: list[float] | None = None,
        **kwargs,
    ):
        if x is None and "X" in kwargs:
            x = kwargs.pop("X")
        if y is None and "Y" in kwargs:
            y = kwargs.pop("Y")
        super().__init__(**kwargs)
        self.X = x
        self.Y = y

    def set_X(self, x):
        self.X = x

    def set_Y(self, y):
        self.Y = y

    def set_data(self, x=None, y=None):
        self.X = x
        self.Y = y

    def to_args(self):
        if self.X is None:
            return tuple()
        if self.Y is None:
            return (self.X,)
        return (self.X, self.Y)

    def to_plot_args(self):
        return self.to_args()


class MarkerEntity(ArchEntity):
    def __init__(
        self,
        x: list[float] | None = None,
        y: list[float] | None = None,
        **kwargs,
    ):
        if x is None and "X" in kwargs:
            x = kwargs.pop("X")
        if y is None and "Y" in kwargs:
            y = kwargs.pop("Y")
        super().__init__(**kwargs)
        self.X = x
        self.Y = y

    def set_X(self, x):
        self.X = x

    def set_Y(self, y):
        self.Y = y

    def set_data(self, x=None, y=None):
        self.X = x
        self.Y = y

    def to_args(self):
        if self.X is None:
            return tuple()
        if self.Y is None:
            return (self.X,)
        return (self.X, self.Y)

    def to_scatter_args(self):
        return self.to_args()


class GraphType(Enum):
    Arch = "arch"
    Curve = "curve"
    Marker = "marker"
    matFigure = "matfigure"
    Else = "else"


class BasicInfo:
    def __init__(
        self,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        figsize: tuple[int | float, int | float] | None = None,
        **kwargs,
    ):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figsize = figsize
        self.kwargs = dict(kwargs)

    def set_title(self, title):
        self.title = title

    def set_xlabel(self, xlabel):
        self.xlabel = xlabel

    def set_ylabel(self, ylabel):
        self.ylabel = ylabel

    def set_figsize(self, figsize):
        self.figsize = figsize

    def set_kwargs(self, **kwargs):
        self.kwargs.update(kwargs)

    def to_dict(self):
        return {
            "title": self.title,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "figsize": self.figsize,
            **self.kwargs,
        }


class GraphInfo:
    def __init__(
        self,
        arch_info: ArchInfo | list[ArchInfo] | None = None,
    ):
        self.arch_info: list[ArchInfo] = []
        if arch_info is not None:
            self.add(arch_info)

    def add(self, arch_info):
        if isinstance(arch_info, ArchInfo):
            self.arch_info.append(arch_info)
        else:
            self.arch_info.extend(arch_info)

    def clear(self):
        self.arch_info.clear()

    def to_list(self):
        return list(self.arch_info)

    def __len__(self):
        return len(self.arch_info)

    def __iter__(self):
        return iter(self.arch_info)

    def __getitem__(self, index):
        return self.arch_info[index]


class GraphData:
    def __init__(
        self,
        graph_info: GraphInfo | ArchInfo | list[ArchInfo] | None = None,
        graph_entity: ArchEntity | list[ArchEntity] | None = None,
    ):
        self.graph_info = self._make_graph_info(graph_info)
        self.graph_entity: list[ArchEntity] = []
        if graph_entity is not None:
            self.add_entity(graph_entity)

    def _make_graph_info(self, graph_info):
        if graph_info is None:
            return GraphInfo()
        if isinstance(graph_info, GraphInfo):
            return graph_info
        return GraphInfo(graph_info)

    def set_graph_info(self, graph_info):
        self.graph_info = self._make_graph_info(graph_info)

    def add_entity(self, graph_entity):
        if isinstance(graph_entity, ArchEntity):
            self.graph_entity.append(graph_entity)
        else:
            self.graph_entity.extend(graph_entity)

    def add(self, graph_entity):
        self.add_entity(graph_entity)

    def clear(self):
        self.graph_entity.clear()

    def to_tuple(self):
        return self.graph_info, self.graph_entity

    def __len__(self):
        return len(self.graph_entity)

    def __iter__(self):
        return iter(self.graph_entity)

    def __getitem__(self, index):
        return self.graph_entity[index]


class Arch:
    def __init__(
        self,
        basic_info: BasicInfo | None = None,
        data: list[tuple[GraphType, GraphData]] | None = None,
    ):
        self.basic_info = basic_info if basic_info is not None else BasicInfo()
        self.data: list[tuple[GraphType, GraphData]] = []
        if data is not None:
            for graph_type, graph_data in data:
                self._append(graph_type, graph_data)

    def set_basic_info(self, basic_info):
        self.basic_info = basic_info

    def _append(self, graph_type, graph_data):
        if isinstance(graph_type, str):
            graph_type = GraphType(graph_type)
        if not isinstance(graph_type, GraphType):
            raise TypeError("graph_type must be GraphType or str")
        if not isinstance(graph_data, GraphData):
            raise TypeError("graph_data must be GraphData")
        self.data.append((graph_type, graph_data))

    def add(self, graph_data):
        self._append(GraphType.Arch, graph_data)

    def add_curve(self, graph_data):
        self._append(GraphType.Curve, graph_data)

    def add_marker(self, graph_data):
        self._append(GraphType.Marker, graph_data)

    def clear(self):
        self.data.clear()

    def to_tuple(self):
        return self.basic_info, self.data

    def to_list(self):
        return list(self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]


# one Arch is one Figure
class ArchPlot:
    def __init__(
        self,
        arch_data: Arch | list[Arch] | None = None,
        figpath: str | None = None,
        modelpath: str | None = None,
    ):
        self.arch_data: list[Arch] = []
        self.figpath = figpath
        self.modelpath = modelpath
        if arch_data is not None:
            self.add(arch_data)

    def add(self, arch_data: Arch | list[Arch] | None = None):
        if arch_data is None:
            return self
        if isinstance(arch_data, Arch):
            self.arch_data.append(arch_data)
            return self
        for arch in arch_data:
            if not isinstance(arch, Arch):
                raise TypeError("arch_data must be Arch or list[Arch]")
            self.arch_data.append(arch)
        return self

    def add_curve(
        self,
        x: list[float | int],
        y: list[float | int],
        arch_info: CurveInfo | list[CurveInfo] | GraphInfo | None = None,
        basic_info: BasicInfo | None = None,
        **kwargs,
    ):
        graph_data = GraphData(
            graph_info=arch_info,
            graph_entity=CurveEntity(x=x, y=y, **kwargs),
        )
        arch = Arch(basic_info=basic_info)
        arch.add_curve(graph_data)
        self.add(arch)
        return arch

    def add_marker(
        self,
        x: list[float | int],
        y: list[float | int],
        arch_info: MarkerInfo | list[MarkerInfo] | GraphInfo | None = None,
        basic_info: BasicInfo | None = None,
        **kwargs,
    ):
        graph_data = GraphData(
            graph_info=arch_info,
            graph_entity=MarkerEntity(x=x, y=y, **kwargs),
        )
        arch = Arch(basic_info=basic_info)
        arch.add_marker(graph_data)
        self.add(arch)
        return arch

    # indices: use indices to indicate what archs should be merged from self.arch_data
    #
    # mindex: different arch has different basicinfo, so we should use mindex(main_index) to
    # explicily point out what arch's basicinfo is final-true basicinfo
    # if mindex == None, the first arch'basicinfo is final-true
    def compose(
        self,
        indices: list[int] | None = None,
        mindex: int | None = None,
    ) -> Arch:
        if not self.arch_data:
            raise ValueError("no arch data to compose")

        selected_indices = list(range(len(self.arch_data))) if indices is None else list(indices)
        if not selected_indices:
            raise ValueError("indices cannot be empty")

        main_index = selected_indices[0] if mindex is None else mindex
        if main_index not in selected_indices:
            raise ValueError("mindex must be included in indices")

        composed = Arch(basic_info=self.arch_data[main_index].basic_info)
        for index in selected_indices:
            arch = self.arch_data[index]
            for graph_type, graph_data in arch:
                composed._append(graph_type, graph_data)
        return composed

    def _get_arch_info(self, graph_info, index):
        if len(graph_info) == 0:
            return None
        if index < len(graph_info):
            return graph_info[index]
        if len(graph_info) == 1:
            return graph_info[0]
        return None

    def _plot_entity(self, axes, graph_type, entity, arch_info):
        plot_kwargs = {}
        if arch_info is not None:
            plot_kwargs.update(arch_info.to_kwargs())
        plot_kwargs.update(entity.to_kwargs())

        if graph_type == GraphType.Curve or isinstance(entity, CurveEntity):
            axes.plot(*entity.to_plot_args(), **plot_kwargs)
            return
        if graph_type == GraphType.Marker or isinstance(entity, MarkerEntity):
            axes.scatter(*entity.to_scatter_args(), **plot_kwargs)
            return
        raise TypeError(f"unsupported graph entity: {type(entity).__name__}")

    def _apply_basic_info(self, axes, basic_info):
        if basic_info.title is not None:
            axes.set_title(basic_info.title)
        if basic_info.xlabel is not None:
            axes.set_xlabel(basic_info.xlabel)
        if basic_info.ylabel is not None:
            axes.set_ylabel(basic_info.ylabel)

    def _group_graph_data(self, arch):
        grouped_data: dict[GraphType, list[GraphData]] = {}
        for graph_type, graph_data in arch:
            grouped_data.setdefault(graph_type, []).append(graph_data)
        return grouped_data

    def _render_arch(self, arch):
        figure_kwargs = dict(arch.basic_info.kwargs)
        if arch.basic_info.figsize is not None:
            figure_kwargs.setdefault("figsize", arch.basic_info.figsize)

        grouped_data = self._group_graph_data(arch)
        if not grouped_data:
            figure, axes = plt.subplots(**figure_kwargs)
            self._apply_basic_info(axes, arch.basic_info)
            figure.tight_layout()
            return figure

        figure, axes = plt.subplots(len(grouped_data), 1, squeeze=False, **figure_kwargs)
        axes = axes.ravel()

        for axes_index, (graph_type, graph_data_list) in enumerate(grouped_data.items()):
            graph_axes = axes[axes_index]
            self._apply_basic_info(graph_axes, arch.basic_info)
            if len(grouped_data) > 1 and arch.basic_info.title is not None:
                graph_axes.set_title(f"{arch.basic_info.title} - {graph_type.value}")

            for graph_data in graph_data_list:
                for index, entity in enumerate(graph_data):
                    arch_info = self._get_arch_info(graph_data.graph_info, index)
                    self._plot_entity(graph_axes, graph_type, entity, arch_info)

            handles, labels = graph_axes.get_legend_handles_labels()
            if handles and labels:
                graph_axes.legend()
        figure.tight_layout()
        return figure

    def to_fig(
        self,
        indices: list[int] | None = None,
        mindex: int | None = None,
    ):
        return self._render_arch(self.compose(indices=indices, mindex=mindex))

    def _resolve_save_path(
        self,
        filename: str | None,
        configured_path: str | None,
        default_dir: str,
        default_filename: str,
    ):
        path = Path(filename or configured_path) if filename or configured_path else Path.cwd() / default_dir
        if path.suffix == "":
            path = path / default_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_load_path(
        self,
        filename: str | None,
        configured_path: str | None,
        default_filename: str,
    ):
        if filename is None and configured_path is None:
            raise ValueError("filename or modelpath must be provided")
        path = Path(filename or configured_path)
        if path.suffix == "":
            path = path / default_filename
        return path

    # save logic:
    # if filename is not None: first-use
    # elif self.figpath/self.modelpath is not None: use it
    # elif the path is current_dictionary/fig or current_dictionary/model
    #
    # the model uses pickle lib to save and load
    def save_fig(self, filename: str | None = None):
        path = self._resolve_save_path(filename, self.figpath, "fig", "arch_plot.png")
        figure = self.to_fig()
        figure.savefig(path)
        plt.close(figure)
        return path

    def save_model(self, filename: str | None = None):
        path = self._resolve_save_path(filename, self.modelpath, "model", "arch_plot.pkl")
        with open(path, "wb") as file:
            pickle.dump(self.arch_data, file)
        return path

    # load logic:
    # filename first, self.modelpath second
    # cannot be use default_path, if all paths are None, raise the exception
    def load_model(self, filename: str | None = None):
        path = self._resolve_load_path(filename, self.modelpath, "arch_plot.pkl")
        with open(path, "rb") as file:
            arch_data = pickle.load(file)
        self.arch_data = []
        self.add(arch_data)
        return self

    def show(self):
        figure = self.to_fig()
        plt.show()
        return figure

    def clear(self):
        self.arch_data.clear()

    def to_list(self):
        return list(self.arch_data)

    def __len__(self):
        return len(self.arch_data)

    def __iter__(self):
        return iter(self.arch_data)

    def __getitem__(self, index):
        return self.arch_data[index]


# make plt's figure obj transTo Arch and _add to self
# def add_figure(self, fig):
#     pass

# return figuric ap
# def to_fig(self):
#     pass


def roc(
    histories,
    title: str | None = None,
    include_baseline: bool = True,
    arch: Arch | None = None,
) -> Arch:
    import numpy as np

    def scalar(value):
        return float(np.asarray(value).ravel()[0])

    def normalize_histories(histories):
        if (
            isinstance(histories, tuple)
            and len(histories) == 2
            and isinstance(histories[0], str)
        ):
            return [histories]
        return histories

    def build_curve(pred_history):
        scores = np.array([scalar(pred) for pred, _ in pred_history])
        labels = np.array([int(scalar(label)) for _, label in pred_history])

        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        if positives == 0 or negatives == 0:
            raise ValueError("ROC requires both positive and negative samples")

        order = np.argsort(-scores, kind="mergesort")
        scores_sorted = scores[order]
        labels_sorted = labels[order]

        true_positives = np.cumsum(labels_sorted == 1)
        false_positives = np.cumsum(labels_sorted == 0)

        distinct = np.r_[np.where(np.diff(scores_sorted))[0], scores_sorted.size - 1]
        tpr = np.r_[0.0, true_positives[distinct] / positives]
        fpr = np.r_[0.0, false_positives[distinct] / negatives]
        auc = float(np.trapezoid(tpr, fpr))
        return fpr.tolist(), tpr.tolist(), auc

    plot_title = "ROC" if title is None else f"ROC {title}"
    if arch is None:
        arch = Arch(BasicInfo(title=plot_title, xlabel="FPR", ylabel="TPR"))
    elif not isinstance(arch, Arch):
        raise TypeError("arch must be Arch or None")
    else:
        if arch.basic_info.title is None:
            arch.basic_info.set_title(plot_title)
        if arch.basic_info.xlabel is None:
            arch.basic_info.set_xlabel("FPR")
        if arch.basic_info.ylabel is None:
            arch.basic_info.set_ylabel("TPR")

    for label, pred_history in normalize_histories(histories):
        fpr, tpr, auc = build_curve(pred_history)
        arch.add_curve(
            GraphData(
                CurveInfo(f"{label} (AUC={auc:.4f})"),
                CurveEntity(x=fpr, y=tpr),
            )
        )

    if include_baseline:
        arch.add_curve(
            GraphData(
                CurveInfo("random", color="gray", style="-."),
                CurveEntity(x=[0.0, 1.0], y=[0.0, 1.0]),
            )
        )

    return arch
