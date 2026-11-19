# GTA VI Trophy Rarity Forecast (Week One)

[![make.fun competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xiaogui917/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xiaogui917/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Probabilistic forecasting engine for predicting the earn rates (5 quantiles: 5%, 25%, 50%, 75%, 95%) of Grand Theft Auto VI PlayStation 5 trophies on Exophase after release, evaluated via mean pinball loss on the logit scale.

---

## Forecasting Methodology

The engine models trophy earn rate evolution as a non-stationary stochastic process governed by tracked player cohort influx and mission progression dynamics:

1. **Logit Scale Modeling**:
   - Trophy earn rates $r \in [0.01, 99.99]$ are transformed to log-odds space:
     $$\text{logit}(r) = \ln\left(\frac{p}{1 - p}\right), \quad p = \frac{r}{100}$$
   - Quantiles $q_\tau$ for $\tau \in \{0.05, 0.25, 0.50, 0.75, 0.95\}$ are evaluated by pinball loss:
     $$L_\tau(y, q) = 2 \max(\tau \cdot d, (\tau - 1) \cdot d), \quad d = \text{logit}(y) - \text{logit}(q)$$

2. **Launch-Night Dilution Dynamics**:
   - Exophase refreshes player counts in discrete batches. At launch night, initial tracking cohorts (~600 players) expand rapidly to thousands of tracked players.
   - For early tutorial trophies (e.g. prologue mission "Gone Fishing"), virtually all new players earn them immediately (~99-100%).
   - For mid/late chapters, platinum, 100% completion, and hard collectibles, new players cannot finish them immediately; the denominator expands rapidly, causing temporary rate dilution.
   - The engine models this using a mixture prior with an adaptive spread that covers both pre-refresh and post-refresh states.

3. **Active Progression & Velocity Damping**:
   - In observation windows with multiple reads and changing rates, the empirical log-odds velocity $v = \frac{d\,\text{logit}(r)}{dt}$ is estimated via variance-weighted linear regression on recent reads.
   - To prevent runaway extrapolation over longer horizons (e.g. 4.5 days to 1 week), progression velocity is exponentially damped with a half-life of 36 hours:
     $$\Delta L = v \cdot \frac{1 - e^{-\gamma \Delta t}}{\gamma}$$
   - The trend is shrunk toward a domain prior regularizing tiers (Platinum, Gold, Silver, Bronze) and mission categories.

4. **Calibrated Spread**:
   - The predictive spread accounts for horizon uncertainty scaled by $\sqrt{\text{days}}$ and the historical residual volatility of each individual trophy, ensuring high coverage and minimal pinball penalty.

---

## Local Execution

The forecaster runs using Python 3.13 standard library only, with zero third-party dependencies and no network access.

```bash
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

### Environment Variables

- `MAKEFUN_INPUT_DIR`: Directory containing `<id>.in` JSON cases (default: `/input`).
- `MAKEFUN_OUTPUT_DIR`: Directory where `<id>.out` forecasts will be written (default: `/output`).

### Output Format

Each `<id>.out` file contains valid JSON meeting the program/v1 schema:

```json
{
  "trophies": [
    {"id": 1, "q": [0.0333, 0.0782, 0.1415, 0.2557, 0.5988]},
    {"id": 2, "q": [99.1142, 99.3678, 99.5, 99.6047, 99.7182]}
  ]
}
```

- Each trophy has 5 finite numbers in $[0, 100]$, strictly non-decreasing, rounded to 4 decimals half-up.
- Output file size remains well below the 65,536-byte limit (~2.8 KB per case).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
