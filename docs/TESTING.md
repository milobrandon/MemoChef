# Testing Guide

## Test Suites

### Unit Tests (92 tests, ~2 min)
```bash
python -m pytest --ignore=tests/stress_test_market_data.py --ignore=tests/randomize_and_test.py --ignore=test_output_randomized -x -q
```

### Playwright UI Tests (10 tests, ~2 min)
Requires the Streamlit app running on localhost:8501.

```bash
# Terminal 1: Start the app
streamlit run app.py --server.headless true --server.port 8501

# Terminal 2: Run tests
python -m pytest tests/test_playwright_ui.py -v
```

#### Setup
```bash
pip install playwright pytest-playwright
python -m playwright install chromium
```

#### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MEMO_CHEF_URL` | `http://localhost:8501` | Streamlit app URL |
| `MEMO_CHEF_USER` | *(required)* | Login username |
| `MEMO_CHEF_PASS` | *(required)* | Login password |
| `HEADED` | (unset) | Set to `1` to watch the browser |

#### What's Tested
- **Login flow**: app loads, bad password rejected, successful login
- **Main UI**: no exceptions, file uploaders present, "Directions for Claude" expander, config profile selectbox, run controls, tabs
- **File upload**: memo + proforma accepted without errors

### Pipeline Verification (no API key needed)
```bash
python -c "
from memo_automator import extract_memo_content, extract_proforma_data, load_config
from memo_chef.slide_generator import extract_deck_profile
cfg = load_config('config.yaml')
memo_content = extract_memo_content('path/to/memo.pptx', cfg)
proforma_data = extract_proforma_data('path/to/proforma.xlsx', cfg)
profile = extract_deck_profile('path/to/memo.pptx', memo_content)
print(f'Memo: {len(memo_content)} chars, Proforma: {len(proforma_data)} chars')
print(f'Deck: {profile.total_slides} slides, font={profile.title_font_name}')
"
```

## CI Workflow
All unit tests run on every push via `.github/workflows/ci.yml`.
Playwright tests are manual (require running Streamlit server + browser).
