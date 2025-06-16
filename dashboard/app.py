from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import umap
from plotly import graph_objects as go
from sklearn.manifold import TSNE

from datasets import TorchvisionDataset
from fvhd import FVHD
from knn import Graph, NeighborConfig, NeighborGenerator

num_points = st.sidebar.slider(
    "Number of points", min_value=1000, max_value=50_000, value=1000, step=1000
)
marker_size = st.sidebar.slider(
    "Marker size", min_value=1, max_value=10, value=3, step=1
)
edge_size = st.sidebar.slider(
    "Edge width", min_value=0.01, max_value=1.0, value=0.1, step=0.01
)

ALGORITHMS = ["FVHD", "umap", "t-SNE"]
embedding_algo = st.sidebar.selectbox("Embedding algorithm", ALGORITHMS)

DATASETS = ["mnist", "fmnist", "qmnist", "kmnist"]
dataset = st.sidebar.selectbox("Dataset", DATASETS)

update_plot = st.sidebar.button("Show plot")

if "plots" not in st.session_state:
    st.session_state.plots = []


@st.cache_data()
def load_data(dataset_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = TorchvisionDataset(dataset_id)
    X = dataset.train_images()
    labels = dataset.train_labels()
    return X, labels


def create_graphs(X: np.ndarray, metric: str = "euclidean", nn: int = 5) -> list[Graph]:
    config = NeighborConfig(metric=metric)
    df = pd.DataFrame(X)
    generator = NeighborGenerator(df=df, config=config)
    graphs = generator.run(nn=nn)
    return graphs


def fit_fvhd_embedding(
    X: np.ndarray,
    graphs: list[Graph],
    nn: int = 3,
    rn: int = 1,
    mutual_neighbors_epochs: int = 20,
) -> np.ndarray:

    fvhd = FVHD(
        n_components=2,
        nn=nn,
        rn=rn,
        c=0.1,
        eta=0.2,
        epochs=3000,
        device="mps",
        velocity_limit=True,
        autoadapt=True,
        mutual_neighbors_epochs=mutual_neighbors_epochs,
    )

    embeddings = fvhd.fit_transform(X, graphs)
    return embeddings


def plot():

    X, y = load_data(dataset)
    X = X[:num_points]
    y = y[:num_points]

    if embedding_algo == "umap":
        X_embedded = umap.UMAP(n_components=2).fit_transform(X)
    elif embedding_algo == "t-SNE":
        X_embedded = TSNE(
            n_components=2, init="pca", learning_rate="auto"
        ).fit_transform(X)
    else:
        graphs = create_graphs(X, metric="euclidean", nn=5)
        X_embedded = fit_fvhd_embedding(
            X, graphs, nn=5, rn=2, mutual_neighbors_epochs=None
        )

    color_map = {
        0: "#e6194b",  # red
        1: "#3cb44b",  # green
        2: "#4363d8",  # blue
        3: "#f58231",  # orange
        4: "#911eb4",  # purple
        5: "#46f0f0",  # cyan
        6: "#f032e6",  # magenta
        7: "#bcf60c",  # lime
        8: "#fabebe",  # pink
        9: "#008080",  # teal
    }

    colors = [color_map[label] for label in y.numpy()]

    node_trace = go.Scatter(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        mode="markers",
        marker=dict(
            size=marker_size,
            color=colors,
        ),
        text=[f"Point {i} - {label}" for i, label in enumerate(y)],
        hoverinfo="text",
        showlegend=False,
    )

    if embedding_algo == "FVHD":
        edge_x = []
        edge_y = []

        for i in range(num_points):
            for j in graphs[0].indexes[i][1:]:
                edge_x += [X_embedded[i, 0], X_embedded[j, 0], None]
                edge_y += [X_embedded[i, 1], X_embedded[j, 1], None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=edge_size, color="lightgray"),
            hoverinfo="skip",
            showlegend=False,
        )

        fig = go.Figure(data=[node_trace, edge_trace])
    else:
        fig = go.Figure(data=[node_trace])

    fig.update_layout(
        title=f"Dataset: {dataset}, Algorithm: {embedding_algo}",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="closest",
    )

    st.session_state.plots.append(fig)



if update_plot:
    plot()

for i, fig in enumerate(st.session_state.plots):
    col1, col2 = st.columns([0.9, 0.1])

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if st.button("X", key=f"del-{i}"):
            st.session_state.plots.pop(i)
            st.rerun()
