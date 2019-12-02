import numpy as np

# This is a collection of utility methods for operating on time.

def get_year_ranges(start_year, end_year):
    # Produces arrays of year data
    # years is an array [args.start_year, ..., arg.send_year]
    # years_dec is years converted to be between 0 and 1
    # num_years is the number of years
    years = np.arange(start_year, end_year)
    years_dec = (years - start_year) / (end_year - start_year)
    num_years = len(years)
    return years, years_dec, num_years


def year2dec(year, start_year, end_year):
    # converts a year to a number between 0 and 1
    return (year - start_year) / (end_year - start_year)


def dec2year(dec, start_year, end_year):
    # converts a number between 0 and 1 to a year
    return dec * (end_year - start_year) + start_year
