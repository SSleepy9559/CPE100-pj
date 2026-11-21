# Build data/sat_table.csv and data/sh_table.csv from the three raw steam-table CSVs
# (University of Colorado Boulder tables supplied by the group). No CoolProp needed.
# Owner: A
#
# Inputs  (searched in data/raw/, data/, then the project root; version numbers in the
#          file names do not matter, e.g. saturated_by_pressure_V1.4.csv is found too):
#   saturated_by_pressure*.csv                    saturation properties, rows by pressure
#   saturated_by_temperature*.csv                 saturation properties, rows by temperature
#   compressed_liquid_and_superheated_steam*.csv  liquid, saturated, vapour, supercritical
# Outputs (written to data/):
#   sat_table.csv            P, Tsat, vf, vg, hf, hg, sf, sg
#   sh_table.csv             P, T, v, h, s, phase
#   steam_table_SOURCE.txt   copyright and licence notice copied from the raw files
#
# Units: P in MPa, T in degrees C, v in m3/kg, h in kJ/kg, s in kJ/(kg.K).
# Run from the project root:  python scripts/make_steam_table.py

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.paths import ROOT, DATA_DIR

# ---------------------------------------------------------------- settings
RAW_DIRS = [DATA_DIR / 'raw', DATA_DIR, ROOT]            # where to look for the raw files
SAT_P_PATTERN = 'saturated_by_pressure*.csv'
SAT_T_PATTERN = 'saturated_by_temperature*.csv'
SH_PATTERN = 'compressed_liquid_and_superheated_steam*.csv'

# The raw files have a few metadata lines first, then the header line starting with these.
SAT_P_HEADER_START = 'p (mpa)'
SAT_T_HEADER_START = 't ('
SH_HEADER_START = 'pressure (mpa)'

# Raw columns are renamed by position (the raw names contain units and the degree sign).
SAT_P_COLS = ['P', 'Tsat', 'vf', 'vg', 'uf', 'ug', 'ufg', 'hf', 'hg', 'hfg', 'sf', 'sg', 'sfg']
SAT_T_COLS = ['Tsat', 'P', 'vf', 'vg', 'uf', 'ug', 'ufg', 'hf', 'hg', 'hfg', 'sf', 'sg', 'sfg']
SH_COLS = ['P', 'T', 'v', 'rho', 'u', 'h', 's', 'phase']

SAT_OUT_COLS = ['P', 'Tsat', 'vf', 'vg', 'hf', 'hg', 'sf', 'sg']     # contract: section 16.5
SH_OUT_COLS = ['P', 'T', 'v', 'h', 's', 'phase']                      # 'phase' is an extra column
SH_VALUE_COLS = ['v', 'rho', 'u', 'h', 's']

PHASE_MAP = {'liquid': 'liquid',
             'saturated liquid': 'sat_liquid',
             'saturated vapor': 'sat_vapor',
             'vapor': 'vapor',
             'supercritical fluid': 'supercritical'}

CROSS_CHECK_TOL = 0.005    # max relative difference between the two saturation tables (0.5 %)

# Design states from the project doc: every one must exist in the tables (P in MPa, T in C).
REQUIRED_PRESSURES = [0.010, 1.2, 15.0]    # P7 (condenser), P6 (extraction), P5 (boiler)
REQUIRED_SUPERHEATED = (15.0, 600.0)       # state 5: boiler exit (P5, T5)


# ---------------------------------------------------------------- reading
def find_raw_file(pattern):
    """Find a raw table file by name pattern.

    Args:
        pattern: glob pattern such as 'saturated_by_pressure*.csv'.

    Returns:
        Path of the match (the last one in name order if there are several versions).

    Raises:
        FileNotFoundError: if no folder in RAW_DIRS holds a matching file.
    """
    for folder in RAW_DIRS:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[-1]
    searched = ', '.join(str(d) for d in RAW_DIRS)
    raise FileNotFoundError(f"No file matching '{pattern}' found in: {searched}")


def read_raw_table(path, header_start, columns):
    """Read one raw CSV: split off the metadata lines and rename the columns.

    Args:
        path: Path of the raw CSV file.
        header_start: lower-case start of the real header line (marks where data begins).
        columns: new column names, one per raw column, in order.

    Returns:
        (table, metadata): DataFrame with the new column names, and the list of metadata
        lines (version, author, copyright, licence, date) found above the header.

    Raises:
        ValueError: if the header line is missing or the column count is wrong.
    """
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    header_index = next((i for i, line in enumerate(lines)
                         if line.strip().lower().startswith(header_start)), None)
    if header_index is None:
        raise ValueError(f"{path.name}: header line starting with '{header_start}' not found")
    metadata = [line.rstrip(', \t') for line in lines[:header_index] if line.strip(', \t')]
    table = pd.read_csv(path, skiprows=header_index, encoding='utf-8-sig',
                        skipinitialspace=True)
    if len(table.columns) != len(columns):
        raise ValueError(f"{path.name}: expected {len(columns)} columns, "
                         f"found {len(table.columns)}")
    table.columns = columns
    return table, metadata


# ---------------------------------------------------------------- building
def build_sat_table(raw):
    """Clean the saturation table (rows by pressure) into the contract format.

    Args:
        raw: DataFrame from read_raw_table with SAT_P_COLS names.

    Returns:
        DataFrame with columns P, Tsat, vf, vg, hf, hg, sf, sg, sorted by P with no
        duplicate pressures (the raw file repeats some rows).
    """
    table = raw[SAT_OUT_COLS].apply(pd.to_numeric, errors='coerce')
    table = table.dropna().drop_duplicates(subset='P')
    return table.sort_values('P').reset_index(drop=True)


def build_sh_table(raw):
    """Clean the compressed-liquid / superheated table into the contract format.

    Rows with missing values are dropped (the raw file has a few corrupted rows at very
    high pressure). Phase names are shortened to: liquid, sat_liquid, sat_vapor, vapor,
    supercritical. Saturated rows are kept as end points; use 'phase' to avoid
    interpolating across the saturation dome.

    Args:
        raw: DataFrame from read_raw_table with SH_COLS names.

    Returns:
        (table, dropped): the cleaned DataFrame (columns P, T, v, h, s, phase, sorted by
        P then T) and a DataFrame of the rows that were dropped.

    Raises:
        ValueError: if an unknown phase name appears.
    """
    table = raw.copy()
    numeric_cols = ['P', 'T'] + SH_VALUE_COLS
    table[numeric_cols] = table[numeric_cols].apply(pd.to_numeric, errors='coerce')
    table['phase'] = table['phase'].astype(str).str.strip().str.lower()

    unknown = set(table['phase']) - set(PHASE_MAP)
    if unknown:
        raise ValueError(f'Unknown phase names in steam table: {sorted(unknown)}')
    table['phase'] = table['phase'].map(PHASE_MAP)

    bad = table[numeric_cols].isna().any(axis=1) | (table['v'] <= 0)
    dropped = table[bad]
    table = table[~bad].drop_duplicates(subset=['P', 'T', 'phase'])
    # Stable sort keeps 'sat_liquid' before 'sat_vapor' at the same P and T.
    table = table.sort_values(['P', 'T'], kind='mergesort').reset_index(drop=True)
    return table[SH_OUT_COLS], dropped


# ---------------------------------------------------------------- checking
def cross_check_sat(sat, sat_by_temp):
    """Compare the two saturation tables with each other.

    Interpolates the by-pressure table to each temperature of the by-temperature table
    and returns the largest relative difference in hf, hg, sf and sg.

    Args:
        sat: DataFrame from build_sat_table.
        sat_by_temp: raw by-temperature DataFrame with SAT_T_COLS names (numeric).

    Returns:
        float, the largest relative difference (0.001 means 0.1 %).
    """
    worst = 0.0
    for col in ['hf', 'hg', 'sf', 'sg']:
        reference = np.interp(sat_by_temp['Tsat'], sat['Tsat'], sat[col])
        other = sat_by_temp[col].to_numpy(dtype=float)
        rel = np.abs(reference - other) / np.maximum(np.abs(other), 1e-9)
        worst = max(worst, float(rel.max()))
    return worst


def validate_tables(sat, sh):
    """Check the finished tables; raise ValueError if anything is wrong.

    Args:
        sat: DataFrame from build_sat_table.
        sh: DataFrame from build_sh_table.

    Returns:
        None. Raises ValueError describing the first problem found.
    """
    if sat.isna().any().any() or sh.isna().any().any():
        raise ValueError('NaN found in a steam table')
    if not sat['P'].is_monotonic_increasing or not sat['Tsat'].is_monotonic_increasing:
        raise ValueError('sat_table: P and Tsat must both increase down the table')
    if not ((sat['vf'] <= sat['vg']).all() and (sat['hf'] <= sat['hg']).all()
            and (sat['sf'] <= sat['sg']).all()):
        raise ValueError('sat_table: liquid values must not exceed vapour values')

    for pressure in REQUIRED_PRESSURES:
        if not np.isclose(sat['P'], pressure).any():
            raise ValueError(f'sat_table has no row for P = {pressure} MPa')
        if not np.isclose(sh['P'], pressure).any():
            raise ValueError(f'sh_table has no row for P = {pressure} MPa')

    p5, t5 = REQUIRED_SUPERHEATED
    state5 = sh[np.isclose(sh['P'], p5) & np.isclose(sh['T'], t5)]
    if state5.empty:
        raise ValueError(f'sh_table has no row for P = {p5} MPa, T = {t5} C (state 5)')


# ---------------------------------------------------------------- writing
def write_source_note(path, notices):
    """Write the copyright and licence notice carried over from the raw files.

    The licence requires its notice to stay with copies of the data.

    Args:
        path: output Path.
        notices: dict {raw file name: list of metadata lines}.

    Returns:
        None (writes a text file).
    """
    parts = ['sat_table.csv and sh_table.csv were generated by scripts/make_steam_table.py',
             'from the raw steam tables listed below. The notices are copied from the files.',
             'Cite the source in the report.', '']
    for name, lines in notices.items():
        parts.append(f'--- {name}')
        parts.extend(lines)
        parts.append('')
    path.write_text('\n'.join(parts), encoding='utf-8')


def main():
    """Read the raw tables, build and check the two output tables, write them to data/."""
    sat_p_path = find_raw_file(SAT_P_PATTERN)
    sat_t_path = find_raw_file(SAT_T_PATTERN)
    sh_path = find_raw_file(SH_PATTERN)

    sat_raw, sat_notice = read_raw_table(sat_p_path, SAT_P_HEADER_START, SAT_P_COLS)
    sat_t_raw, sat_t_notice = read_raw_table(sat_t_path, SAT_T_HEADER_START, SAT_T_COLS)
    sh_raw, sh_notice = read_raw_table(sh_path, SH_HEADER_START, SH_COLS)

    sat = build_sat_table(sat_raw)
    sh, dropped = build_sh_table(sh_raw)
    validate_tables(sat, sh)

    difference = cross_check_sat(sat, sat_t_raw.apply(pd.to_numeric, errors='coerce'))
    if difference > CROSS_CHECK_TOL:
        raise ValueError(f'Saturation tables disagree by {difference:.2%} '
                         f'(limit {CROSS_CHECK_TOL:.1%})')

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sat.to_csv(DATA_DIR / 'sat_table.csv', index=False)
    sh.to_csv(DATA_DIR / 'sh_table.csv', index=False)
    write_source_note(DATA_DIR / 'steam_table_SOURCE.txt',
                      {sat_p_path.name: sat_notice, sat_t_path.name: sat_t_notice,
                       sh_path.name: sh_notice})

    print(f'sat_table.csv: {len(sat)} rows, P {sat["P"].min():.6g} to {sat["P"].max():.6g} MPa')
    print(f'sh_table.csv : {len(sh)} rows, {sh["P"].nunique()} pressures, '
          f'P {sh["P"].min():g} to {sh["P"].max():g} MPa')
    print(f'Dropped {len(dropped)} corrupted row(s) from the superheated table: '
          + (', '.join(f'(P={r.P:g}, T={r.T:g})' for r in dropped.itertuples()) or 'none'))
    print(f'Saturation tables agree within {difference:.3%} (limit {CROSS_CHECK_TOL:.1%})')

    p5, t5 = REQUIRED_SUPERHEATED
    state5 = sh[np.isclose(sh['P'], p5) & np.isclose(sh['T'], t5)].iloc[0]
    print(f'Check state 5 (P={p5:g} MPa, T={t5:g} C): h = {state5["h"]} kJ/kg, '
          f's = {state5["s"]} kJ/(kg.K)  (textbook h about 3583)')


if __name__ == '__main__':
    main()
