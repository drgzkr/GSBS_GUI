# GSBS GUI

An interactive GUI for running and exploring results of the
[Greedy State Boundary Search (GSBS)](https://github.com/lgeerligs/statesegmentation)
algorithm, which segments fMRI timeseries data into discrete neural states.

![screenshot placeholder](https://github.com/user-attachments/assets/74192d8d-0f8e-45fd-94a0-848ba51d13e4)

---

## Requirements

- Python 3.10+
- `tkinter` — included in the Python standard library.
  On Ubuntu/Debian it ships as a separate system package:
  ```bash
  sudo apt-get install python3-tk
  ```

---

## Installation

### 1. Clone

```bash
git clone https://github.com/drgzkr/GSBS_GUI.git
cd GSBS_GUI
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

| Platform | Command |
|----------|---------|
| Linux / macOS | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd.exe | `.venv\Scripts\activate.bat` |

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python run_gsbs_gui.py
```

### Workflow

1. **Browse** to a `.npy` file containing a 2D array of shape
   `(n_timepoints, n_channels)`.
2. Click **Load ROI Data** to load and preview the correlation matrix and
   raw timeseries.
3. Set **kmax**, **finetune**, and the **Statewise detection** toggle.
4. Click **Run GSBS** — runs in a background thread; the UI stays responsive.
5. When done, drag the **k slider** to explore all solutions from k=1 to kmax:
   - T-dist curve with the selected k marked
   - Correlation matrix with state boundary boxes overlaid
   - Raw voxel timeseries with boundary lines
   - State-averaged timeseries with boundary lines
6. To skip re-running: browse to a saved `.npy` GSBS object and click
   **Load GSBS Object**.
7. Click **Save GSBS Object** to persist the fitted result.

### Input data format

```python
import numpy as np
# shape: (timepoints, channels/voxels)
data = np.load("my_roi_data.npy")   # must be 2D
```

---

## Development

### Install dev dependencies

```bash
pip install -r requirements-dev.txt
```

### Run tests

```bash
pytest tests/
```

Tests are fully headless — no display server or `xvfb` needed.
`conftest.py` patches `tkinter` and `matplotlib`'s TkAgg backend at the
`sys.modules` level before any test module is collected.

#### With coverage report

```bash
pytest tests/ --cov=run_gsbs_gui --cov-report=term-missing
```

### Project layout

```
run_gsbs_gui.py          main application (GSBSApp class)
requirements.txt         runtime dependencies
requirements-dev.txt     development / test dependencies
tests/
    conftest.py          headless mock setup (runs before collection)
    fixtures.py          shared fixtures and fake GSBS object factory
    test_state_timeseries.py   _state_timeseries numpy logic
    test_validation.py         load_roi_data / load_gsbs_object validation
    test_gsbs_done.py          _gsbs_done / _gsbs_error thread callbacks
    test_slider.py             slider state and _on_slider callback
```

### CI (GitHub Actions example)

```yaml
- name: Run tests
  run: |
    pip install -r requirements-dev.txt
    pytest tests/ --cov=run_gsbs_gui --cov-report=xml
```

No system `python3-tk` package is needed in CI because `conftest.py`
intercepts the `import tkinter` statement via `sys.modules` before it
reaches the C extension.

---

## Known limitations / TODO

- Stimulus image viewer (before/after boundary frames) not yet re-implemented
  in the reworked version.
- macOS figure DPI scaling not specifically handled.
