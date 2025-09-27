combined_data_2025-09.csv

Data generated as follows:

We compiled an updated dataset of DOC transmission (flux out / flux in) and WRT (units years), including only field-based data (i.e. excluding bioassay data) and adding in additional data from rivers and streams. We included:
- The C2016 (Catalan et al., 2016) field-based data for rivers, “catchments” (large Swedish systems of chained lakes and rivers) and estuaries. Where the C2016 data included multiple measurements from a single site, we replaced these with site means, resulting in a single data point per site. In this way, all study sites have the same weight in the statistical analysis.
- For lake data, we used the E2017 (Evans et al., 2017) data (see E2017 SI for rationale and a description of issues with the C2016 lake data).
- We then compiled additional DOC transmission and WRT data from rivers and streams through a literature review.

We focused on waterbodies with net DOC removal, as the primary aim here was to explore net DOC decay.

Columns in the csv include:
- Site identifier
- Site type
- Latitude and Longitude (WGS84 decimal degrees). Note that for some of the river/stream sites, these were not reported and are approximate.
- tau: water retention time in years
- mo_mi: DOC transmission, i.e. flux out / flux in. In the case of lakes, this is often calculated as concentration in the outflow or the lake / concentration in the inflow, assuming a fully mixed lake and that outflow discharge = inflow discharge.
- Ref: Data reference. For data points taken from E2017, these are given only as the identifiers provided in E2017
- Source: Data source, either C2017, E2017 or this study
- Country.