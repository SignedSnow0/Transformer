//! Command-line entry point for the BERT examples.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -- [--cpu] [--spam] [DATA_TSV] [STEPS]
//! ```
//!
//! By default this runs the masked-language-model example on
//! `data/eng-it-sample.tsv` for 2,000 steps. With `--spam` it runs the SMS
//! ham/spam classifier on `data/sms-spam-sample.tsv` for 250 steps. The
//! implementation of each example lives in `src/bert/examples/`.
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
