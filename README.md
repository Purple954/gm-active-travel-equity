# GM Active Travel Equity

**[View Dashboard](./dashboard.html)**

Does cycling and walking infrastructure investment reach the communities that need it most, or does it concentrate in areas that are already well-off? As a resident of Greater Manchester, I wanted to answer that question with data rather than assumption. This project uses open government data - cycling infrastructure locations, neighbourhood deprivation scores, and Census travel behaviour - to examine whether active travel provision across the ten GM boroughs varies systematically with deprivation, and what that means for transport inequality across the region.

---

## Research question

**Does active travel infrastructure provision across Greater Manchester vary systematically with neighbourhood deprivation, and what does this mean for travel inequality across the region?**

The analysis works at Lower Super Output Area (LSOA) level, which is the smallest geography for which both deprivation data and Census travel data are published. I use the Cycling Infrastructure Database as the measure of cycling supply.

---

## Data sources

All data is openly available from UK government sources. No registration or API key is required for any of the sources.

| Dataset | Publisher | Format | Direct link |
|---|---|---|---|
| OpenStreetMap cycling infrastructure | OpenStreetMap contributors (ODbL) | GeoPackage (via osmnx/Overpass API) | [openstreetmap.org](https://www.openstreetmap.org) |
| Index of Multiple Deprivation 2019 (File 7) | MHCLG | CSV | [assets.publishing.service.gov.uk](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/845345/File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv) |
| Census 2021 TS061 - Method of travel to workplace | ONS via Nomis API | CSV | [nomisweb.co.uk/datasets/c2021ts061](https://www.nomisweb.co.uk/datasets/c2021ts061) |
| LSOA (Dec 2021) boundaries | ONS Open Geography Portal | GeoJSON | [geoportal.statistics.gov.uk](https://geoportal.statistics.gov.uk/datasets/ons::lsoa-dec-2021-boundaries-full-clipped-ew-bgc/about) |
| Local Authority District boundaries | ONS Open Geography Portal | GeoJSON | [geoportal.statistics.gov.uk](https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2021-uk-bfc/about) |

The analysis covers the ten Greater Manchester boroughs: Bolton, Bury, Manchester, Oldham, Rochdale, Salford, Stockport, Tameside, Trafford, and Wigan (ONS LAD codes E08000001-E08000010).

> **Note on cycling data source**: the DfT Cycling Infrastructure Database (CID) was retired by Ordnance Survey in 2024 and is no longer publicly downloadable. This project uses OpenStreetMap as an equivalent open source. OSM has good coverage of cycling infrastructure across Greater Manchester and is updated continuously by local contributors.

---

## Methodology

**1. Filtering to Greater Manchester.** LSOA boundaries and OSM cycling features are filtered to the ten GM boroughs via a spatial join with ONS LAD boundary polygons. All spatial operations run in British National Grid (EPSG:27700) so distance and area calculations are in metres and km².

**2. Infrastructure density.** Cycling infrastructure is pulled from OpenStreetMap using osmnx, covering ways tagged `highway=cycleway`, `cycleway=lane`, `cycleway=track`, and related tags. These are clipped to each LSOA polygon and their total length summed. Dividing by LSOA area gives infrastructure density in metres per km², which is comparable across LSOAs of different sizes.

**3. Deprivation.** IMD 2019 scores and deciles are joined to each LSOA by LSOA code. Quintiles are derived from deciles (quintile 1 = the most deprived 20% of LSOAs nationally). IMD 2019 uses 2011 LSOA codes, so I apply the ONS 2011-to-2021 best-fit lookup before joining.

**4. Active travel mode share.** Census 2021 TS061 gives counts of workers aged 16+ by usual travel method per LSOA. I calculate active travel share as (cycling + walking) / total workers, expressed as a percentage.

**5. OLS regression.** Two models: a baseline with only IMD score and log-transformed infrastructure density as predictors, and a full model adding borough fixed effects to absorb unobserved borough-level factors. Both use HC3 heteroskedasticity-robust standard errors.

**6. Visualisation.** Four static charts using matplotlib and seaborn: infrastructure density by deprivation quintile (box plot), deprivation vs active travel mode share (scatter with OLS line), infrastructure density distribution by borough, and cycling vs walking share by quintile. An interactive Folium choropleth map has toggleable layers for infrastructure density, deprivation quintile, and active travel mode share.

---

## Key findings

**Active travel is highest in the most deprived areas - and infrastructure does not explain it.** Q1 (most deprived) has a mean active travel mode share of 12.5%, falling to 4.9% in Q5. Cycling infrastructure density also skews toward more deprived areas (Q1 mean: 1,054 m/km², Q5 mean: 655 m/km²), but this largely reflects the denser urban fabric of inner-city areas rather than deliberate investment.

**Walking drives the deprivation gradient, not cycling.** Q1 LSOAs average 10.6% walking mode share against 3.8% in Q5. The strong IMD-walking correlation (Spearman rho = 0.71, p < 0.001) points to constrained transport choices rather than good provision: lower car ownership and cost barriers to public transport make walking the default in the most deprived areas.

**OLS regression confirms both deprivation and infrastructure are significant predictors.** Each additional IMD point is associated with a 0.16 percentage point increase in active travel share (95% CI: 0.15-0.17, p < 0.001). Log infrastructure density is also positively significant (coef = 0.09, p < 0.001), meaning denser cycling infrastructure correlates with slightly higher active travel even after controlling for deprivation. Borough fixed effects lift R² from 0.36 to 0.42; Manchester and Salford have notably higher active travel rates than other boroughs at equivalent deprivation levels.

**The infrastructure-deprivation relationship is weakly positive, not negative.** OSM data shows marginally higher cycling infrastructure density in more deprived LSOAs (Pearson r = 0.09). This reflects the city-centre geography of GM: Manchester and Salford inner areas are both deprived and reasonably well-supplied with cycling infrastructure. The equity question is less about total provision and more about quality and directness - something this analysis cannot measure from OSM tags alone.

---

## Policy implications

The finding that walking mode share is highest in the most deprived GM neighbourhoods should not be read as those areas being well-served. It most likely reflects residents walking because the alternatives - owning a car or affording regular public transport - are out of reach. The fact that infrastructure density is slightly higher in deprived areas does not resolve this: quantity without quality or network connectivity does not translate into safe, attractive cycling conditions. Transport investment appraisal in Greater Manchester should weight equity of access explicitly, not just aggregate route utility or cost-benefit ratios based on current trip volumes. Areas where people are already walking in large numbers despite poor conditions represent the highest-value targets for infrastructure improvement.

---

## Limitations

- OSM cycling infrastructure coverage depends on volunteer contributor activity. Urban core areas in GM are generally well-mapped, but some suburban or residential cycle routes may be missing or inconsistently tagged. Infrastructure quality (surface condition, physical separation from traffic) is not captured in OSM tags at this level of analysis.
- IMD 2019 uses 2011 LSOA boundaries. I handle boundary changes through the ONS best-fit lookup, but for LSOAs that were substantially redrawn between 2011 and 2021 there will be some mismatch.
- Census 2021 travel-to-work captures usual method of travel, not actual trips, and excludes people not in work. Areas with high unemployment or economic inactivity may look like they have lower active travel simply because fewer residents are counted.
- OLS assumes a linear relationship and can be sensitive to high-leverage LSOAs in the upper tail of infrastructure density. The log transformation reduces this but does not fix it entirely.

---

## How to reproduce

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/gm-active-travel-equity.git
cd gm-active-travel-equity
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download raw data

```bash
python src/ingest.py
```

All five sources download automatically. No manual steps are needed.

### 3. Process

```bash
python src/clean.py
```

Outputs `data/processed/gm_lsoa_analysis.gpkg` and `data/processed/gm_lsoa_analysis.csv`. Works without the CID - infrastructure density will show as NaN until that file is present.

### 4. Analyse

```bash
python src/analyse.py
```

Prints regression tables to stdout and saves `outputs/regression_summary.txt`.

### 5. Visualise

```bash
python src/visualise.py
```

Saves four PNG figures to `outputs/figures/` and the interactive HTML map to `outputs/maps/`.

### 6. Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

Runs the full pipeline end-to-end with inline outputs.

---

## Repository structure

```
gm-active-travel-equity/
├── data/
│   ├── raw/              # gitignored - populated by ingest.py
│   └── processed/        # generated by clean.py
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── ingest.py         # data download
│   ├── clean.py          # spatial processing and joining
│   ├── analyse.py        # OLS regression and summary statistics
│   └── visualise.py      # figures and interactive map
├── outputs/
│   ├── figures/          # PNG charts
│   └── maps/             # interactive HTML map
├── requirements.txt
├── .gitignore
└── README.md
```
