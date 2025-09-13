# Data Dictionary

This document outlines the variables collected for the DEX valuation study.

## Daily Metrics (per protocol per day)

| Variable | Description | Units |
| --- | --- | --- |
| `market_cap_circulating` | Circulating market capitalisation of the token at 00:00 UTC | USD |
| `volume_24h` | Total trading volume in USD over the past 24 hours | USD |
| `fees_24h` | Total user-paid fees over the past 24 hours | USD |
| `revenue_24h` | Protocol revenue (portion of fees kept by the protocol) over the past 24 hours | USD |
| `tvl` | Total value locked in USD | USD |
| `active_users_24h` | Number of unique wallet addresses interacting with the DEX over the past 24 hours | count |
| `transactions_24h` | Number of swaps/trades executed over the past 24 hours | count |

## Static Metrics

| Variable | Description | Units |
| --- | --- | --- |
| `token_holders` | Total unique holders of the protocol token | count |
| `governance_proposals_30d` | Number of governance proposals over the last 30 days (normalized per protocol) | count |
| `token_distribution` | Concentration metrics such as Gini coefficient, Herfindahl index, whale concentration | proportion |
| `chain_deployment` | Number of active blockchain networks the DEX is deployed on | count |
| `token_age_days` | Age of the token in days since launch | days |

## Feature Engineering

Refer to the documentation in `src/preprocessing.py` for details on feature transformations and calculated variables.
