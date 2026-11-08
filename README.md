# xStocks movers forecast

Entry for make.fun's `xstocks-movers-forecast-01`: for each of the 5 session days from 2026-11-10 to 2026-11-16,
the probability of each of 9 tokenized Nasdaq names at each place of the day's return order, and each name's
return in 42 bins. `node main.ts` (Node 22, standard library only) reads `/input` (or `$MAKEFUN_INPUT_DIR`) and
writes `/output/forecast.json` (or `$MAKEFUN_OUTPUT_DIR`). Every draw is seeded, so two runs give the same bytes.

## Model
- **Vol.** A prior daily vol per name from two years of the underlying shares' daily closes (0.4 x EWMA variance
  with lambda 0.97 + 0.6 x the 250-day variance), embedded as constants. The history's ~22 hourly bars only nudge
  it: variance x (market-wide realized/prior ratio)^0.3 x (the name's own ratio over the market's)^0.1. One day
  of on-chain hourly bars says little about one name (on the practice day the pools' realized vol was 0.6x and
  1.8x the shares' for NVDA and TSLA), but their geometric mean matched.
- **Tails.** Student-t with 6 degrees of freedom, with the bins taken from the exact t CDF. Against the normal it
  gains about 0.13 a day in the backtest.
- **Order.** 400,000 seeded draws of a multivariate t with a shared chi-square scale and the names' daily
  correlation matrix (last 120 sessions, shrunk 30% to its mean). Market-wide days move every name together.
- **Day 1.** The history already holds the move from Monday's close TWAP to 18:00 New York. Day 1 is centred
  on 0.9 of it (in two years of hourly data, the 16:00 to 18:00 move carried into the next close with a slope
  of 0.9 to 1.0), with 3% of a day's variance taken off for each hour already seen.
- **Later days.** Zero drift, variance pulled 5% a day toward the long-run level, x1.1 after a weekend.

## Evidence
A port of the scorer reproduces the house baseline's published practice score (-7.527502); this entry scores
-7.143593 on that day. The backtest (every forecast date over the last ~240 sessions, 5 days each, with the
shares' pre/post-market hourly realized variance standing in for the hourly bars): **-41.84 a day against
-43.34** for a proxy of the baseline (normal, sd from one day's hourly variance, independent names). That's
about +7.5 over a 5-day final, and both halves of the dates agree.
