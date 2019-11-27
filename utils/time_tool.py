import numpy as np

# This is a collection of utility methods for operating on time.

def get_year_ranges(args):
    # Produces arrays of year data
    # years is an array [args.start_year, ..., arg.send_year]
    # years_dec is years converted to be between 0 and 1
    # num_years is the number of years
    years = np.arange(args.start_year, args.end_year)
    years_dec = (years - args.start_year) / (args.end_year - args.start_year)
    num_years = len(years)
    return years, years_dec, num_years


def year2dec(year, args):
    # converts a year to a number between 0 and 1
    return (year - args.start_year) / (args.end_year - args.start_year)


def dec2year(dec, args):
    # converts a number between 0 and 1 to a year
    return dec * (args.end_year - args.start_year) + args.start_year
