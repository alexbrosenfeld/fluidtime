import numpy as np

def get_year_ranges(args):
    years = np.arange(args.start_year, args.end_year)
    years_dec = (years - args.start_year) / (args.end_year - args.start_year)
    num_years = len(years)
    return years, years_dec, num_years

def year2dec(year, args):
    return (year - args.start_year) / (args.end_year - args.start_year)

def dec2year(dec, args):
    return dec*(args.end_year - args.start_year) + args.start_year