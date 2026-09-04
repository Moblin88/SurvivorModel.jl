"""
    _first_nonmissing(v)

Return the first non-`missing` value in `v`, or `missing` if every element is
`missing`. Used to pull drive-level fields (which nflverse repeats on every
play of a drive but sometimes omits on administrative rows such as kickoffs,
timeouts, or the synthetic "GAME" row) out of a group of plays.
"""
function _first_nonmissing(v)
    for x in v
        ismissing(x) || return x
    end
    return missing
end

"""
    _parse_time_of_possession(s)

Parse an nflverse `drive_time_of_possession` string formatted as `"M:SS"` (or
`"MM:SS"`) into a `Dates.Second` duration (a `Period`, not a time-of-day).
This supports arithmetic and comparisons directly (e.g. `top > Minute(5)`),
and can be pretty-printed with `Dates.canonicalize` (e.g. `"4 minutes, 1
second"`). Returns `missing` if `s` is `missing`.
"""
function _parse_time_of_possession(::Missing)
    return missing
end
function _parse_time_of_possession(s::AbstractString)
    return Second(Time(s, dateformat"M:S") - Time(0))
end

"""
    _summarize_drive(sub::AbstractDataFrame)

Reduce the play-by-play rows belonging to a single drive (already grouped by
`game_id` and `fixed_drive`, in their original chronological order) down to a
single drive-level summary `NamedTuple`. The `game_id`/`fixed_drive` group
keys themselves are not included here since `combine` (see
[`summarize_drives`](@ref)) automatically prepends them to the result.

Notes on the individual fields:
- `posteam`/`defteam` are taken from the *first* play of the drive, since the
  possessing/defending team at the start of a drive is not necessarily the
  same as at the end (e.g. a "Opp touchdown" drive, where the defense returns
  a turnover or blocked kick for a touchdown). `posteam_home`/`defteam_home`
  are boolean indicators for whether each of those teams is the home team.
- `yardline_100` is the distance (in yards) to the opponent's end zone at the
  start of the drive, i.e. nflverse's own `yardline_100` convention (taken
  from the first play where it is present).
- `yards_gained` is the sum of every play's `yards_gained` within the drive
  (`missing` plays are skipped; if all plays are missing, the result is
  `missing`).
- `home_spread_change` is computed from the change in `total_home_score` and
  `total_away_score` from the first to the last play of the drive, so it
  reflects the *net* effect of the drive on the scoreboard from the home
  team's perspective (e.g. a made field goal is `+3`, an opponent pick-six
  with a good PAT is `-7`).
"""
function _summarize_drive(sub::AbstractDataFrame)
    home_team = sub.home_team[1]

    posteam = _first_nonmissing(sub.posteam)
    defteam = _first_nonmissing(sub.defteam)
    posteam_home = ismissing(posteam) ? missing : posteam == home_team
    defteam_home = ismissing(defteam) ? missing : defteam == home_team
    drive_result = _first_nonmissing(sub.fixed_drive_result)
    time_of_possession = _parse_time_of_possession(_first_nonmissing(sub.drive_time_of_possession))
    yardline_100 = _first_nonmissing(sub.yardline_100)

    yards_vals = collect(skipmissing(sub.yards_gained))
    yards_gained = isempty(yards_vals) ? missing : sum(yards_vals)

    home_score_change = sub.total_home_score[end] - sub.total_home_score[begin]
    away_score_change = sub.total_away_score[end] - sub.total_away_score[begin]
    home_spread_change = home_score_change - away_score_change

    return (;
        posteam, defteam, posteam_home, defteam_home,
        drive_result, time_of_possession, yardline_100, yards_gained, home_spread_change,
    )
end

"""
    summarize_drives(pbp::AbstractDataFrame) -> DataFrame

Given a play-by-play `DataFrame` as returned by `NFLData.load_pbp`, return a
`DataFrame` with one row per drive, containing:

- `game_id`, `fixed_drive`: identify the game and the drive within that game
  (following nflverse's own naming/typing conventions).
- `posteam`, `defteam`: the possessing and defending teams at the *start* of
  the drive.
- `posteam_home`, `defteam_home`: whether `posteam`/`defteam` is the home
  team (or `missing` if unknown).
- `drive_result`: the (fixed) drive outcome, e.g. `"Touchdown"`, `"Punt"`,
  `"Turnover"`, `"Field goal"`, `"Opp touchdown"`, etc.
- `time_of_possession`: duration of possession during the drive, as a
  `Dates.Second` (or `missing`).
- `yardline_100`: distance to the opponent's end zone at the start of the
  drive (nflverse's `yardline_100` convention: 0 = opponent's goal line, 100 =
  own goal line).
- `yards_gained`: net yards gained over the course of the drive.
- `home_spread_change`: change in the score margin from the home team's
  perspective (positive means the home team gained ground).

Some plays are missing data (e.g. the synthetic "GAME" row, timeouts, or
administrative plays), so fields that are constant across a drive are pulled
from the first play where they are present, and `yards_gained` skips missing
plays entirely.
"""
function summarize_drives(pbp::AbstractDataFrame)
    grouped = groupby(pbp, [:game_id, :fixed_drive]; skipmissing=true)
    return combine(_summarize_drive, grouped)
end

"""
    load_drive_pbp(args...)

Load NFL drive-by-drive summary data. Accepts the same arguments as
`NFLData.load_pbp` (e.g. `seasons`), loads the underlying play-by-play data,
and reduces it to one row per drive via [`summarize_drives`](@ref).
"""
function load_drive_pbp(args...)
    return summarize_drives(load_pbp(args...))
end
