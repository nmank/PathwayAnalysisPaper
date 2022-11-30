"""Module for plotting results."""


# imports
from pandas import DataFrame
import plotly.graph_objects as go

# functions
def score_violin_plot(score_frame: DataFrame) -> None:
    """
    Generates a plotly boxplot figure for a dataframe of scores of shape
    (n_runs, n_scores)
    """
    # initialize Plotly figure
    fig = go.Figure()
    
    # generate box plot for each feature
    for score in score_frame:
        fig.add_trace(go.Violin(y=score_frame[score].values, name=score, visible='legendonly', box_visible=True, meanline_visible=True))
    
    return fig
    