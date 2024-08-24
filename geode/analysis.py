from .core import *


def combine_data_handlers(handler_names, database, db_table):
    # Combines the data from various datahandlers
    data = []
    for name in handler_names:
        handler = DataHandler(name=name, db_table=db_table,
                              database=database, load=True)
        df = handler.values
        df["name"] = name
        data.append(df)

    # Stacks dataframes
    data = pd.concat(data, join="outer", ignore_index=True)

    return data


def combine_data_series(handler_names, database, db_table, mean=True, mean_col=None):
    # Gets data list
    data_list = []
    for handler_list in handler_names:
        try:
            handler_list = list(handler_list)
        except TypeError:
            handler_list = [handler_list]

        df = combine_data_handlers(
            handler_list, database=database, db_table=db_table)
        df = df.drop("name", axis=1)

        if mean:
            if mean_col is None:
                df = df.mean()
            else:
                df = mean_column(df, mean_col)

        data_list.append(df)

    return data_list


def series_to_csv(data_list, path_root, names):
    for df, name in zip(data_list, names):
        df.to_csv(os.path.join(path_root, name))


def get_tests(root, test_num, use_root=True):
    # Gets range of test names from a root
    names = []

    if use_root:
        # Starts with root
        names = [root]
        test_num -= 1

    # Adds numeric tests
    for num in range(test_num):
        names.append("{}-{}".format(root, num))

    return names


def mean_column(df, column):
    # Gets mean of every value of a specific column

    # Blank dataframe to add means to
    means = pd.DataFrame(columns=df.columns)

    # Loops through every value of column
    for val in df[column].unique():
        # Appends mean at column value
        val_mean = df[df[column] == val].drop(column, axis=1).mean()
        val_mean[column] = val
        means = means.append(val_mean, ignore_index=True)

    # Returns means
    return means


def reduce_series(df_list, func):
    reduce_df = pd.DataFrame()
    for df in df_list:
        reduce_df = reduce_df.append(func(df), ignore_index=True)
    return reduce_df

# Plotting functions


def get_train_fig(suptitle, metric_titles, ax_labels):
    # Creates figure and axes
    fig, axes = plt.subplots(1, len(metric_titles))

    fig.suptitle(suptitle, fontsize=20)
    for num, (title, label) in enumerate(zip(metric_titles, ax_labels)):
        axes[num].set_ylabel(label)
        axes[num].set_xlabel("Training Epoch")
        axes[num].set_title(title)

    return fig, axes

def get_test_fig(suptitle, metric_titles, ax_labels):
    # Creates figure and axes
    fig, axes = plt.subplots(1, len(metric_titles))

    fig.suptitle(suptitle, fontsize=20)
    for num, (title, label) in enumerate(zip(metric_titles, ax_labels)):
        axes[num].set_ylabel(label)
        axes[num].set_title(title)

    return fig, axes


def plot_train_series(fig, axes, train_list, test_list, labels, metrics, colors=None, use="both", linestyle="solid", legend=True, legend_loc="upper center"):
    # Colors are list of Nones by default
    colors = [None]*len(labels) if colors is None else colors

    for num, metric in enumerate(metrics):
        for label, color, train_data, test_data in zip(labels, colors, train_list, test_list):
            if use != "val":
                axes[num].plot(train_data["epoch"], train_data[metric], color=color, linestyle=linestyle,
                            label=label)
            if use != "train":
                val_label = None if use == "both" else label
                axes[num].plot(train_data["epoch"], train_data[f"val_{metric}"], color=color, linestyle=linestyle,
                            marker='v', label=val_label)

            if use == "both":
                axes[num].fill_between(train_data["epoch"], train_data[metric],
                                    train_data[f"val_{metric}"], color=color, alpha=0.2)
            
            test_style = 'dashed' if linestyle == "solid" else linestyle
            axes[num].hlines(test_data[metric], train_data["epoch"].min(), train_data["epoch"].max(),
                        colors=color,
                        linestyles=test_style)
            axes[num].set_xticks(train_data["epoch"])

        if legend:
            if num == 1:
                # Creates model legend
                model_legend=axes[num].legend(loc=legend_loc)
                axes[num].add_artist(model_legend)

def train_proxy_legend(axes):
    # Creates line legend with proxy plots
    proxies = [
        plt.plot([], color="black", label="train"),
        plt.plot([], color="black", marker="v", label="validation"),
        plt.plot([], color="black", linestyle="dashed", label="test")
    ]

    proxies = [proxy[0] for proxy in proxies]
    proxy_legend = axes[1].legend(handles=proxies, loc=(0.7, 0.07))
    axes[1].add_artist(proxy_legend)

def plot_test_whisker(test_list, labels, metrics, metric_titles=None, ax_labels=None, suptitle=None, colors=None):
    # Metric titles are list of Nones by default
    metric_titles = [None]*len(metrics) if metric_titles is None else metric_titles

    # Y axis labels are metrics by default
    ax_labels = metrics if ax_labels is None else ax_labels

    # Colors are list of Nones by default
    colors = [None]*len(labels) if colors is None else colors

    # Creates figure and axes
    fig, axes = plt.subplots(1, len(metrics))
    fig.tight_layout(pad=2)
    # Makes axes a list
    try:
        axes = list(axes)
    except TypeError:
        axes = [axes]
    
    if not suptitle is None:
        fig.suptitle(suptitle, fontsize=20)

    for num, (title, ax_label, metric) in enumerate(zip(metric_titles, ax_labels, metrics)):
        metric_data = [test[metric] for test in test_list]
        if not title is None:
            axes[num].set_title(title)
        box1 = axes[num].boxplot(metric_data, patch_artist=True, showmeans=True, labels=labels, widths=0.4)
        axes[num].set_xticklabels(labels, rotation=-20)
        axes[num].set_ylabel(ax_label)

        # Sets colors
        for patch, color in zip(box1['boxes'], colors):
            patch.set_facecolor(color)

    return fig, axes
