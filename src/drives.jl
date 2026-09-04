"""
    _parse_time_of_possession(s)

Parse an nflverse `drive_time_of_possession` string formatted as `"M:SS"` (or
`"MM:SS"`) into a `Dates.CompoundPeriod` expressed in mixed minutes/seconds
(e.g. `"4 minutes, 1 second"`), via `Dates.canonicalize`. This still supports
arithmetic and comparisons directly (e.g. `top > Minute(5)`). Returns
`missing` if `s` is `missing`.
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
  (`missing` plays are skipped; `0` if there are no non-`missing` plays).
- `home_spread_change` is computed from the change in `total_home_score` and
  `total_away_score` from the first to the last play of the drive, so it
  reflects the *net* effect of the drive on the scoreboard from the home
  team's perspective (e.g. a made field goal is `+3`, an opponent pick-six
  with a good PAT is `-7`).
"""
function _summarize_drive(sub::AbstractDataFrame)
    home_team = sub.home_team[1]

    # Fields are constant across a drive but sometimes omitted on
    # administrative rows (kickoffs, timeouts, "END QUARTER" markers), so
    # pull the first non-`missing` value. This assumes at least one play in
    # the drive has each field present, which holds for every real drive
    # since `summarize_drives` already drops drives with no real plays.
    posteam = first(skipmissing(sub.posteam))
    defteam = first(skipmissing(sub.defteam))
    posteam_home = posteam == home_team
    defteam_home = defteam == home_team
    drive_result = first(skipmissing(sub.fixed_drive_result))
    time_of_possession = _parse_time_of_possession(first(skipmissing(sub.drive_time_of_possession)))
    yardline_100 = first(skipmissing(sub.yardline_100))

    yards_gained = sum(skipmissing(sub.yards_gained); init=0)

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
  `Dates.CompoundPeriod` in mixed minutes/seconds (or `missing`).
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

nflverse also inserts synthetic bookkeeping rows with no real play (e.g. the
"GAME", "END GAME", and "END QUARTER N" marker rows, identifiable by having
`play_type === missing`) to mark the start/end of a game, half, or quarter.
These normally merge into an adjacent real drive, but occasionally end up
isolated in their own `fixed_drive` group (e.g. when the game/half ends on
the very last play). Such groups have no real plays at all
(`all(ismissing, play_type)`), so they are dropped entirely before
summarizing, rather than appearing as a drive with all-`missing` fields.
"""
function summarize_drives(pbp::AbstractDataFrame)
    grouped = groupby(pbp, [:game_id, :fixed_drive]; skipmissing=true)
    real_drives = filter(sub -> !all(ismissing, sub.play_type), grouped)
    return combine(_summarize_drive, real_drives)
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
