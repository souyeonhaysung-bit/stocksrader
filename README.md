# Real-time Momentum Radar (minimal seed)

This repo contains a minimal, runnable seed for the Momentum Radar:
- Ingest prices (yfinance)
- Compute multi-horizon returns
- Rank by region (toy mapping)
- Plot a quick heatmap

Run:
```bash
pip install -r requirements.txt
python run_pipeline.py
markdown
코드 복사

**requirements.txt**
pandas
numpy
yfinance
matplotlib

bash
코드 복사
> (처음엔 최소로. 나중에 streamlit/plotly/altair 등 추가)

**config/hierarchy_map.csv**  *(티커→국가/지역 샘플; 필요에 맞게 교체)*
```csv
symbol,country,region,continent,sector,industry
SPY,USA,North America,Americas,Multi,Multi
EWY,Korea,East Asia,Asia,Multi,Multi
EPI,India,South Asia,Asia,Multi,Multi
EWJ,Japan,East Asia,Asia,Multi,Multi
EWO,Austria,Western Europe,Europe,Multi,Multi
