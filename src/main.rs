//! Command-line entry point for the BERT examples.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -- [--cpu] [--spam] [--d-model N] [--d-ff N]
//!                          [--n-heads N] [--layers N] [--max-len N]
//!                          [--dropout P] [DATA_TSV] [STEPS]
//! ```
//!
//! By default this runs the masked-language-model example on
//! `data/eng-it-full.tsv` for 20,000 steps. With `--spam` it runs the SMS
//! ham/spam classifier on `data/sms-spam-full.tsv` for 1,000 steps. Both
//! examples default to `d_model=128`, `d_ff=256`, 4 heads, 2 layers,
//! `max_len=64`, and `dropout=0.1`; `--max-len` must be at least four and
//! `--n-heads` must divide `--d-model`. Corpora can be regenerated with
//! `python scripts/prepare_datasets.py`; see `data/README.md`. The implementation
//! of each example lives in `src/bert/examples/`.
//!
//! The tensor math runs on WebGPU when an adapter is available and falls back
//! to the CPU backend otherwise. Pass `--cpu` to force the CPU.

mod autograd;
mod bert;
mod training;
mod transformer;

use bert::examples::{run_mlm, run_spam};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime, is_wgpu_available};
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let spam_mode = env::args().any(|argument| argument == "--spam");
    let force_cpu = env::args().any(|argument| argument == "--cpu");
    let use_wgpu = !force_cpu && is_wgpu_available();

    if !force_cpu && !use_wgpu {
        eprintln!("No WebGPU adapter found; falling back to the CPU backend.");
    }

    match (spam_mode, use_wgpu) {
        (true, true) => run_spam::<WgpuRuntime>(&WgpuDevice::new(0)),
        (true, false) => run_spam::<CpuRuntime>(&CpuDevice::new()),
        (false, true) => run_mlm::<WgpuRuntime>(&WgpuDevice::new(0)),
        (false, false) => run_mlm::<CpuRuntime>(&CpuDevice::new()),
    }
}
