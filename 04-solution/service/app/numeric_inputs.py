"""Numeric CLI domains; stdlib only so invalid options fail before any work."""
import argparse
import math


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("нужно целое число > 0")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("нужно целое число >= 0")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("нужно конечное число >= 0")
    return number


def unit_interval(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("нужно конечное число от 0 до 1 включительно")
    return number


def validate_rerank_params(k1, k2, lam):
    if k1 < 1 or k2 < 1 or not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("REID_RERANK_K1/K2 должны быть >= 1, "
                         "REID_RERANK_LAMBDA — конечным числом от 0 до 1")
