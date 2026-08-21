"""Deterministic result aggregation, intervals, plots, and tables."""

from vla_fewshot.reporting.collect import collect_rollouts
from vla_fewshot.reporting.plots import write_cost_curve_svg
from vla_fewshot.reporting.tables import write_report_tables

__all__ = ["collect_rollouts", "write_cost_curve_svg", "write_report_tables"]
