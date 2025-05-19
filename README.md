# NHL-assist-value-compared-to-goals

This repository contains the code I used to evaluate how valuable primary and secondary assists are relative to goals in the NHL for a blog post I wrote. The methodology was inspired by Tyrel Stoakes and his [post](https://statsbystokes.wordpress.com/2014/08/16/. I developed a custom Goals Created (GC) metric and converted it to a rate stat per 60 minutes of ice time (GC/60). The goal was to find the optimal weights for primary and secondary assists that maximize the predictive power of GC/60 across future seasons. I used R² (coefficient of determination) as the performance metric.

For each TOI threshold the code:

- Filters out players who didn't meet the minimum TOI across all seasons
- Performs a grid search, testing all possible combinations of primary and secondary assists
- For each combination, R² was calculated to measure how well GC/60 in the first four seasons predicts GC/60 in last two seasons
- Visualizes the results with, scatter plots, heat maps, and line graphs

All of the NHL player data that I used came from [NaturalStatTrick](https://www.naturalstattrick.com)

## Check it Out
Below you can view the blog that goes into details on this experiment and results:

[The Blog Post](https://analyticswithavery.com/blog/1)
