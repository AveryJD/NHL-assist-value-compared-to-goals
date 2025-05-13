# NHL-assist-value-compared-to-goals

This repository contains the code I used to evaluate how valuable primary and secondary assists are relative to goals in the NHL for a blog post I wrote. Inspired by Tyrel Stoakes and his [post](https://statsbystokes.wordpress.com/2014/08/16/what-is-the-objective-value-of-an-assist/) from his website Stats by Stoakes, I developed a custom "Goals Created" (GC) metric. The goal was to find the optimal weights for primary and secondary assists that maximize the predictive power of GC/60 across future seasons. I used R² (coefficient of determination) as the performance metric.

For each assist weight combination and TOI threshold, the code:

- Filters out players who didn’t meet the minimum TOI across all seasons.
- Calculates GC/60 for each player.
- Uses a linear regression model using historical GC/60 to predict GC/60 in later seasons.
- Evaluates model accuracy using average R² over two future seasons.
- Visualizes the results with, scatter plots, heat maps, and line graphs

All of the NHL player data that I used came from [NaturalStatTrick](https://www.naturalstattrick.com)

## Check it Out
Below you can view the blog that goes into details on this experiment:

The blog post [Coming Soon] found on [my website](https://analyticswithavery.com).
