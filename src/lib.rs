mod lpc;
mod plugin;

pub use lpc::LpcProcessor;
pub use plugin::LpcPlugin;

nih_plug::nih_export_vst3!(crate::LpcPlugin);
