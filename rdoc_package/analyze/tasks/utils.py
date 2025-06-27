"""Utility functions for task analysis."""

from typing import List, Optional
import polars as pl

def get_metrics(df: pl.DataFrame, group_by: Optional[List[str]] = None) -> pl.DataFrame:
    """Calculate basic metrics from task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.
        group_by (Optional[List[str]]): List of column names to group by.

    Returns:
        pl.DataFrame: DataFrame with calculated metrics.
    """
    # Calculate accuracy using correct_trial (only for trials with responses)
    accuracy = df.group_by(group_by).agg(
        pl.col('correct_trial').filter(pl.col('rt').is_not_null()).mean().alias('accuracy')
    )

    # Calculate omission rate using RT (trials where no response was given)
    omission_rate = df.group_by(group_by).agg(
        pl.col('rt').is_null().mean().alias('omission_rate')
    )

    # Calculate reaction time (only for trials with valid responses)
    rt = df.group_by(group_by).agg(
        pl.col('rt').filter(pl.col('rt').is_not_null()).mean().alias('rt')
    )

    # Join all metrics
    if group_by:
        return accuracy.join(omission_rate, on=group_by).join(rt, on=group_by)
    else:
        # When not grouping, we can just concatenate the columns
        return pl.concat([
            accuracy.select('accuracy'),
            omission_rate.select('omission_rate'),
            rt.select('rt')
        ], how='horizontal')

def organize_metrics(
    metrics: pl.DataFrame, group_by: Optional[List[str]] = None
) -> pl.DataFrame:
    """Organize metrics into a long format with 'metric' and 'value' columns.

    Args:
        metrics (pl.DataFrame): DataFrame containing metrics.
        group_by (Optional[List[str]]): List of column names that were used for grouping.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    def create_metric_column(group_by_cols):
        if not group_by_cols:
            return pl.col('metric')
        
        # Create metric name by concatenating group values
        metric_parts = []
        for col in group_by_cols:
            metric_parts.append(pl.col(col).cast(pl.Utf8))
        
        return pl.concat_str(metric_parts, separator='_')

    # Melt the dataframe to get metric and value columns
    melted = metrics.melt(
        id_vars=group_by if group_by else [],
        value_vars=['accuracy', 'omission_rate', 'rt'],
        variable_name='metric_type',
        value_name='value'
    )

    # Create the metric column
    if group_by:
        melted = melted.with_columns(
            metric=create_metric_column(group_by) + pl.lit('_') + pl.col('metric_type')
        )
    else:
        melted = melted.with_columns(
            metric=pl.col('metric_type')
        )

    # Drop the group_by columns and metric_type column
    if group_by:
        melted = melted.drop(group_by)
    melted = melted.drop('metric_type')

    return melted 