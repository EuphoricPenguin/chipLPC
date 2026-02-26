use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, EguiState};
use egui_extras;
use egui;
use std::sync::{Arc, Mutex};

use crate::lpc::{LpcProcessor, MAX_BUFFER_SIZE};

fn rate_index_to_hz(index: f32) -> f32 {
    let idx = index.round() as i32;
    match idx {
        0 => 8000.0,
        1 => 11025.0,
        2 => 22050.0,
        _ => 44100.0,
    }
}

fn frame_rate_to_hold_period(index: f32) -> usize {
    let idx = index.round() as i32;
    match idx {
        0 => usize::MAX,
        1 => 16,
        2 => 8,
        3 => 4,
        4 => 2,
        _ => 1,
    }
}

#[derive(Params)]
struct LpcParams {
    // Visible knobs - these are exposed for automation
    #[id = "order"]
    pub order: IntParam,
    #[id = "rate"]
    pub rate: FloatParam,
    #[id = "pitch"]
    pub pitch: FloatParam,
    #[id = "tracking"]
    pub tracking: FloatParam,
    
    // Hidden parameters (not exposed - no #[id] attribute)
    // These use sensible defaults and are not visible in the GUI
    buffer_size: IntParam,
    frame_rate: FloatParam,
    detune: FloatParam,
    noise: FloatParam,
    mix: FloatParam,
    glottal: BoolParam,
    preemphasis: BoolParam,
}

impl Default for LpcParams {
    fn default() -> Self {
        Self {
            // Visible knobs
            // Order: 10-100 (LPC order - higher = more detail)
            order: IntParam::new("Order", 18, IntRange::Linear { min: 10, max: 100 })
                .with_unit("coefs"),
            
            // Rate: 8/11/22/44 kHz (discrete values, click to cycle)
            // Default: 8 kHz (index 0)
            rate: FloatParam::new("Rate", 0.0, FloatRange::Linear { min: 0.0, max: 3.0 })
                .with_step_size(1.0)
                .with_unit("kHz"),
            
            // Pitch: -24 to +24 semitones
            pitch: FloatParam::new("Pitch", 0.0, FloatRange::Skewed { min: -24.0, max: 24.0, factor: 1.0 })
                .with_unit("st")
                .with_smoother(SmoothingStyle::Exponential(0.5)),
            
            // Tracking: 0-100%
            // Default: 100%
            tracking: FloatParam::new("Tracking", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit("%")
                .with_smoother(SmoothingStyle::Exponential(0.5)),
            
            // Hidden parameters with fixed defaults
            buffer_size: IntParam::new("Buffer Size", 2048, IntRange::Linear { min: 512, max: 4096 })
                .with_unit("samples"),
            frame_rate: FloatParam::new("Frame Rate", 5.0, FloatRange::Linear { min: 0.0, max: 5.0 })
                .with_step_size(1.0),
            detune: FloatParam::new("Detune", 0.0, FloatRange::Linear { min: 0.0, max: 1200.0 })
                .with_unit("cents"),
            noise: FloatParam::new("Noise", 0.0, FloatRange::Linear { min: -100.0, max: 100.0 })
                .with_unit("%"),
            mix: FloatParam::new("Mix", 100.0, FloatRange::Linear { min: 0.0, max: 100.0 })
                .with_unit("%"),
            glottal: BoolParam::new("Glottal Pulse", false),
            preemphasis: BoolParam::new("Preemphasis", false),
        }
    }
}

pub struct LpcPlugin {
    params: Arc<LpcParams>,
    editor_state: Arc<EguiState>,
    sample_rate: f32,
    lpc: LpcProcessor,
    in_buffer: Vec<f32>,
    out_buffer: Vec<f32>,
    dry_buffer: Vec<f32>,
    buffer_pos: usize,
    frame_counter: usize,
    process_frame: bool,
    held_pitch: f32,
    held_power: f32,
    // Pre-allocated buffers for real-time processing (avoids allocations in audio thread)
    process_buffer: Vec<f32>,
    coef_buffer: Vec<f32>,
    // Input level for LED meter (shared with UI via Mutex)
    input_level: Arc<Mutex<f32>>,
    // Silence detection - consecutive silent frames counter
    silent_frames: usize,
}

impl Default for LpcPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(LpcParams::default()),
            editor_state: EguiState::from_size(900, 600),
            sample_rate: 44100.0,
            lpc: LpcProcessor::new(),
            in_buffer: vec![0.0; MAX_BUFFER_SIZE],
            out_buffer: vec![0.0; MAX_BUFFER_SIZE],
            dry_buffer: vec![0.0; MAX_BUFFER_SIZE],
            buffer_pos: 0,
            frame_counter: 0,
            process_frame: true,
            held_pitch: 0.0,
            held_power: 0.001,
            // Pre-allocate buffers to avoid real-time allocations
            process_buffer: vec![0.0; MAX_BUFFER_SIZE],
            coef_buffer: vec![0.0; 128], // Max order
            // Input level for LED meter (shared with UI)
            input_level: Arc::new(Mutex::new(0.0)),
            // Silence detection counter
            silent_frames: 0,
        }
    }
}

impl Plugin for LpcPlugin {
    const NAME: &'static str = "chip_lpc";
    const VENDOR: &'static str = "EupgoricPenguin";
    const URL: &'static str = "https://penguin.house";
    const EMAIL: &'static str = "euphoricpenguin@protonmail.com";
    const VERSION: &'static str = "0.1.0";

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        // Clone the input level Arc for the UI thread
        let input_level = self.input_level.clone();
        
        // Pre-load images for consistent rendering
        let bg_image = egui::include_image!("../include/BG.png");
        let led_image = egui::include_image!("../include/BG LED Lit.png");
        
        create_egui_editor(
            self.editor_state.clone(),
            self.params.clone(),
            |egui_ctx, _egui_state| {
                egui_extras::install_image_loaders(egui_ctx);
            },
            move |egui_ctx, setter, params| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE.fill(egui::Color32::TRANSPARENT))
                    .show(egui_ctx, |ui| {
                        // Get the available size and use it for all layers
                        let available = ui.available_rect_before_wrap();
                        let size = available.size();
                        
                        // Background - scale to fill the available space
                        ui.put(
                            egui::Rect::from_min_size(egui::pos2(0.0, 0.0), size),
                            egui::Image::new(bg_image.clone())
                                .fit_to_exact_size(size),
                        );
                        
                        // LED layer - between BG and knobs
                        let current_level = input_level.try_lock()
                            .map(|l| *l)
                            .unwrap_or(0.0);
                        
                        let alpha = (current_level * 15.0).clamp(0.0, 1.0);
                        
                        if alpha > 0.01 {
                            ui.put(
                                egui::Rect::from_min_size(egui::pos2(0.0, 0.0), size),
                                egui::Image::new(led_image.clone())
                                    .fit_to_exact_size(size)
                                    .tint(egui::Color32::from_rgba_unmultiplied(
                                        255, 255, 255, (alpha * 255.0) as u8,
                                    )),
                            );
                        }
                        
                        // Get the knob images for layered knob
                        let knob_base_shadow = egui::include_image!("../include/Knob Base.png");
                        let knob_top = egui::include_image!("../include/Knob Top.png");
                        let knob_top_shadow = egui::include_image!("../include/Knob Top Shadow.png");
                        
                        // Knob positions in 2x2 grid
                        let knob_size = 275.0;
                        let spacing_x = 230.0;
                        let spacing_y = 180.0;
                        let grid_start = egui::pos2(300.0 - spacing_x, 220.0 - spacing_y / 2.0);
                        
                        for i in 0..4 {
                            let knob_pos = grid_start + egui::vec2(
                                (i % 2) as f32 * spacing_x,
                                (i / 2) as f32 * spacing_y,
                            );
                            
                            let knob_rect = egui::Rect::from_min_size(knob_pos, egui::vec2(knob_size, knob_size));
                            
                            let params_ref: &LpcParams = &*params;
                            
                            // Single interact for both click and drag - use hover to set cursor
                            let knob_response = ui.interact(knob_rect, egui::Id::new(("knob", i)), egui::Sense::click_and_drag());
                            
                            // Set vertical resize cursor when hovering over knobs
                            if knob_response.hovered() {
                                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                            }
                            
                            // Context menu on right-click
                            knob_response.context_menu(|ui| {
                                // Label showing current value
                                let label = match i {
                                    0 => format!("Order: {:.0}", params.order.value()),
                                    1 => {
                                        let r = params.rate.value().round() as i32;
                                        match r {
                                            0 => "Rate: 8 kHz".to_string(),
                                            1 => "Rate: 11 kHz".to_string(),
                                            2 => "Rate: 22 kHz".to_string(),
                                            _ => "Rate: 44 kHz".to_string(),
                                        }
                                    },
                                    2 => format!("Pitch: {:+.0} st", params.pitch.value()),
                                    3 => format!("Tracking: {:.0}%", params.tracking.value()),
                                    _ => "".to_string(),
                                };
                                ui.label(label);
                                ui.separator();
                                
                                // Reset to default button
                                if ui.button("Reset to Default").clicked() {
                                    match i {
                                        0 => setter.set_parameter(&params_ref.order, 18),
                                        1 => setter.set_parameter(&params_ref.rate, 0.0),
                                        2 => setter.set_parameter(&params_ref.pitch, 0.0),
                                        3 => setter.set_parameter(&params_ref.tracking, 100.0),
                                        _ => {}
                                    }
                                    ui.close_menu();
                                }
                                
                                // Use a nested menu for entering values
                                ui.menu_button("Enter Value...", |ui| {
                                    let (min_val, max_val) = match i {
                                        0 => (10.0, 100.0),      // Order
                                        1 => (0.0, 3.0),         // Rate (discrete)
                                        2 => (-24.0, 24.0),      // Pitch
                                        3 => (0.0, 100.0),       // Tracking
                                        _ => (0.0, 100.0),
                                    };
                                    
                                    // Get current value as default
                                    let current_val: f32 = match i {
                                        0 => params.order.value() as f32,
                                        1 => params.rate.value(),
                                        2 => params.pitch.value(),
                                        3 => params.tracking.value(),
                                        _ => 0.0,
                                    };
                                    
                                    let text_id = egui::Id::new(("knob_input", i));
                                    
                                    // CORRECT: Retrieve the string *inside* the closure and return the string itself
                                    let mut text = ui.data_mut(|d| {
                                        d.get_temp::<String>(text_id.clone())
                                            .unwrap_or_else(|| format!("{:.0}", current_val))
                                    });
                                    
                                    ui.add(egui::TextEdit::singleline(&mut text).desired_width(80.0));
                                    
                                    // CORRECT: Insert the modified string back *inside* the closure
                                    ui.data_mut(|d| d.insert_temp(text_id, text.clone()));
                                    
                                    ui.horizontal(|ui| {
                                        if ui.button("OK").clicked() {
                                            let value: f32 = text.parse().unwrap_or(current_val);
                                            let clamped = value.clamp(min_val, max_val);
                                            match i {
                                                0 => setter.set_parameter(&params_ref.order, clamped as i32),
                                                1 => setter.set_parameter(&params_ref.rate, clamped.round()),
                                                2 => setter.set_parameter(&params_ref.pitch, clamped),
                                                3 => setter.set_parameter(&params_ref.tracking, clamped),
                                                _ => {}
                                            }
                                            // CORRECT: Remove the string from memory *inside* the closure
                                            ui.data_mut(|d| d.remove::<String>(text_id.clone()));
                                            ui.close_menu();
                                        }
                                        if ui.button("Cancel").clicked() {
                                            // CORRECT: Remove the string from memory *inside* the closure
                                            ui.data_mut(|d| d.remove::<String>(text_id.clone()));
                                            ui.close_menu();
                                        }
                                    });
                                    
                                    // Show range hint
                                    let hint = match i {
                                        0 => "10-100",
                                        1 => "0-3 (0=8k,1=11k,2=22k,3=44k)",
                                        2 => "-24 to +24",
                                        3 => "0-100",
                                        _ => "",
                                    };
                                    ui.label(egui::RichText::new(hint).small());
                                });
                            });
                            
                            // Handle left-click for Rate knob to cycle values
                            if knob_response.clicked() && i == 1 {
                                let current_rate = params.rate.value();
                                let next_rate = ((current_rate.round() + 1.0) % 4.0).clamp(0.0, 3.0);
                                setter.set_parameter(&params_ref.rate, next_rate);
                            }
                            
                            // Handle drag for all knobs
                            if knob_response.drag_started() {
                                match i {
                                    0 => setter.begin_set_parameter(&params_ref.order),
                                    1 => setter.begin_set_parameter(&params_ref.rate),
                                    2 => setter.begin_set_parameter(&params_ref.pitch),
                                    3 => setter.begin_set_parameter(&params_ref.tracking),
                                    _ => {}
                                }
                            }
                            
                            if knob_response.dragged() {
                                // Dragging UP gives negative Y delta, invert for value increase
                                let delta = -knob_response.drag_delta().y;
                                
                                match i {
                                    // Rate uses discrete steps (0, 1, 2, 3) - use larger direct steps
                                    1 => {
                                        let current_rate = params.rate.value();
                                        // Each 50 pixels of drag = 1 step
                                        let steps = (delta / 50.0).round() as i32;
                                        if steps != 0 {
                                            let new_rate = (current_rate + steps as f32).clamp(0.0, 3.0);
                                            setter.set_parameter(&params_ref.rate, new_rate);
                                        }
                                    },
                                    // Order, Pitch, Tracking use normalized values
                                    _ => {
                                        let sensitivity = 0.005;
                                        let current_normalized = match i {
                                            0 => params.order.unmodulated_normalized_value(),
                                            2 => params.pitch.unmodulated_normalized_value(),
                                            3 => params.tracking.unmodulated_normalized_value(),
                                            _ => 0.5,
                                        };
                                        
                                        let new_normalized = (current_normalized + delta * sensitivity).clamp(0.0, 1.0);
                                        
                                        match i {
                                            0 => setter.set_parameter_normalized(&params_ref.order, new_normalized),
                                            2 => setter.set_parameter_normalized(&params_ref.pitch, new_normalized),
                                            3 => setter.set_parameter_normalized(&params_ref.tracking, new_normalized),
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            
                            if knob_response.drag_stopped() {
                                match i {
                                    0 => setter.end_set_parameter(&params_ref.order),
                                    1 => setter.end_set_parameter(&params_ref.rate),
                                    2 => setter.end_set_parameter(&params_ref.pitch),
                                    3 => setter.end_set_parameter(&params_ref.tracking),
                                    _ => {}
                                }
                            }
                            
                            // Read the unmodulated value for visual UI feedback
                            let knob_value = match i {
                                0 => params.order.unmodulated_normalized_value(),
                                1 => params.rate.unmodulated_normalized_value(),
                                2 => params.pitch.unmodulated_normalized_value(),
                                3 => params.tracking.unmodulated_normalized_value(),
                                _ => 0.5,
                            };
                            
                            // Layer 1: Knob Base and Shadow (bottom, static)
                            ui.put(
                                knob_rect,
                                egui::Image::new(knob_base_shadow.clone())
                                    .fit_to_exact_size(egui::vec2(knob_size, knob_size)),
                            );
                            
                            // Layer 2: Knob Top (middle, rotates - this is what the user interacts with)
                            // Rotate around center using radians
                            let angle_deg = -135.0 + knob_value * 270.0;
                            let angle_rad = angle_deg.to_radians();
                            
                            // Create a rotated image - fit_to_exact_size ensures consistent sizing during rotation
                            let rotatable_image = egui::Image::new(knob_top.clone())
                                .fit_to_exact_size(egui::vec2(knob_size, knob_size))
                                .rotate(angle_rad, egui::Vec2::splat(0.5));
                            
                            ui.put(knob_rect, rotatable_image);
                            
                            // Layer 3: Knob Top Shadow (top, static - creates depth effect)
                            ui.put(
                                knob_rect,
                                egui::Image::new(knob_top_shadow.clone())
                                    .fit_to_exact_size(egui::vec2(knob_size, knob_size)),
                            );
                        }
                    });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;
        self.reset();
        true
    }

    fn reset(&mut self) {
        self.in_buffer.fill(0.0);
        self.out_buffer.fill(0.0);
        self.dry_buffer.fill(0.0);
        self.buffer_pos = 0;
        self.frame_counter = 0;
        self.process_frame = true;
        self.held_pitch = 0.0;
        self.held_power = 0.001;
        self.lpc.reset_states();
        if let Ok(mut level) = self.input_level.lock() {
            *level = 0.0;
        }
        self.silent_frames = 0;
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let order = self.params.order.value() as usize;
        let buffer_size = (self.params.buffer_size.value() as usize).clamp(512, MAX_BUFFER_SIZE);
        
        let rate = self.params.rate.value();
        let internal_sample_rate = rate_index_to_hz(rate);
        let frame_rate = self.params.frame_rate.value();
        let pitch_shift = self.params.pitch.value();
        let tracking = self.params.tracking.value();
        let detune = self.params.detune.value();
        let noise = self.params.noise.value() / 100.0;
        let mix = self.params.mix.value() / 100.0;
        let glottal = self.params.glottal.value();
        let preemphasis = self.params.preemphasis.value();
        
        let hold_period = frame_rate_to_hold_period(frame_rate);
        let sample_hold_period = (44100.0 / internal_sample_rate).round() as usize;
        
        // Calculate input RMS for LED level meter and silence detection
        let mut level_sum = 0.0f32;
        let mut level_count = 0usize;
        let mut input_rms = 0.0f32;
        
        for (_block_offset, mut block) in buffer.iter_blocks(64) {
            for sample_in_block in 0..block.samples() {
                let b = self.buffer_pos % buffer_size;
                
                let input_sample = if let Some(ch) = block.get(0) {
                    ch[sample_in_block]
                } else {
                    0.0
                };
                
                // Accumulate for RMS calculation
                level_sum += input_sample * input_sample;
                level_count += 1;
                
                self.dry_buffer[b] = input_sample;
                
                if self.buffer_pos > 0 && b == 0 {
                    self.frame_counter += 1;
                    self.process_frame = self.frame_counter >= hold_period;
                    if self.process_frame {
                        self.frame_counter = 0;
                    }
                }
                
                if b == 0 && self.process_frame {
                    self.process_chunk_internal(
                        buffer_size,
                        order,
                        preemphasis,
                        pitch_shift,
                        tracking,
                        detune,
                        noise,
                        glottal,
                        sample_hold_period,
                    );
                }
                
                let output_raw = self.out_buffer[b];
                let wet = output_raw.max(-10.0).min(10.0);
                let dry = self.dry_buffer[b].max(-10.0).min(10.0);
                let output = dry * (1.0 - mix) + wet * mix;
                
                for ch in block.iter_mut() {
                    ch[sample_in_block] = output;
                }
                
                self.in_buffer[b] = input_sample;
                self.buffer_pos = b + 1;
            }
        }
        
        // Update the input level for the LED meter and detect silence
        if level_count > 0 {
            input_rms = (level_sum / level_count as f32).sqrt();
            if let Ok(mut level) = self.input_level.lock() {
                // Smooth the level to prevent flickering
                *level = (*level) * 0.7 + input_rms * 0.3;
            }
        }
        
        // Silence detection - if input is very low for multiple frames, force silence
        const SILENCE_THRESHOLD: f32 = 0.001;
        const SILENCE_FRAMES_TO_STOP: usize = 10;
        
        if input_rms < SILENCE_THRESHOLD {
            self.silent_frames += 1;
        } else {
            self.silent_frames = 0;
        }
        
        // If we've had enough consecutive silent frames, reset the synthesis state
        if self.silent_frames >= SILENCE_FRAMES_TO_STOP {
            self.held_pitch = 0.0;
            self.held_power = 0.001;
            self.lpc.reset_states();
        }

        ProcessStatus::Normal
    }
}

impl LpcPlugin {
    fn process_chunk_internal(
        &mut self,
        buffer_size: usize,
        order: usize,
        preemphasis: bool,
        pitch_shift: f32,
        tracking: f32,
        detune: f32,
        noise: f32,
        glottal: bool,
        sample_hold_period: usize,
    ) {
        if buffer_size < 512 || order < 1 || order > buffer_size / 2 {
            for sample in self.out_buffer[..buffer_size].iter_mut() {
                *sample = 0.0;
            }
            return;
        }
        
        // Use pre-allocated buffer to avoid real-time allocations
        self.process_buffer[..buffer_size].copy_from_slice(&self.in_buffer[..buffer_size]);
        let input = &mut self.process_buffer[..buffer_size];
        
        if preemphasis {
            self.lpc.preemphasis(input, 0.5);
        }

        // Use pre-allocated coefficient buffer
        self.coef_buffer[..order].fill(0.0);
        let coefs = &mut self.coef_buffer[..order];
        let (power, pitch) = self.lpc.analyze(input, coefs, order);

        let mut analyzed_pitch = pitch;
        let mut analyzed_power = power;

        if !analyzed_power.is_finite() || analyzed_power <= 0.0 {
            analyzed_power = 0.001;
        }
        if !analyzed_pitch.is_finite() || analyzed_pitch <= 0.0 {
            analyzed_pitch = 0.0;
        }

        let mut final_pitch: f32;
        let mut final_power: f32;
        
        // Ensure minimum tracking of 1% to always have some output
        let effective_tracking = tracking.max(1.0);
        
        if analyzed_pitch > 0.0 {
            if effective_tracking >= 100.0 {
                final_pitch = analyzed_pitch;
                final_power = analyzed_power;
                self.held_pitch = analyzed_pitch;
                self.held_power = analyzed_power;
            } else {
                final_pitch = self.held_pitch * (1.0 - effective_tracking / 100.0) + analyzed_pitch * (effective_tracking / 100.0);
                final_power = self.held_power * (1.0 - effective_tracking / 100.0) + analyzed_power * (effective_tracking / 100.0);
                
                // Always update held values (even at low tracking)
                self.held_pitch = self.held_pitch * 0.7 + analyzed_pitch * 0.3;
                self.held_power = self.held_power * 0.7 + analyzed_power * 0.3;
            }
        } else {
            final_pitch = self.held_pitch;
            final_power = self.held_power * 0.95;
        }

        if !final_pitch.is_finite() || final_pitch <= 0.0 {
            final_pitch = 0.0;
        }
        if !final_power.is_finite() || final_power <= 0.0 {
            final_power = 0.001;
        }

        if final_pitch > 0.0 {
            let pitch_mult = 2.0f32.powf(pitch_shift / 12.0);
            
            if pitch_mult.is_finite() && pitch_mult > 0.0 {
                final_pitch /= pitch_mult;
            }
            
            if !final_pitch.is_finite() || final_pitch <= 0.0 || final_pitch > buffer_size as f32 {
                final_pitch = buffer_size as f32 / 4.0;
            }
            
            if pitch > 0.0 && pitch.is_finite() {
                final_power = final_power * (pitch / final_pitch);
            }
            
            if !final_power.is_finite() || final_power > 100.0 || final_power <= 0.0 {
                final_power = 0.001;
            }
        }

        let mut noise_adjusted_pitch = final_pitch;
        if noise < -0.01 {
            if final_pitch == 0.0 && self.held_pitch > 0.0 {
                noise_adjusted_pitch = self.held_pitch;
            }
        } else if noise > 0.01 {
            // Do NOT set this to 0.0 if LpcProcessor uses it for period math!
            // Use a low baseline frequency instead to prevent divide-by-zero
            noise_adjusted_pitch = 10.0;
        }

        // Defensive guard: Ensure we don't accidentally exceed the coef_buffer size
        let safe_order = order.clamp(1, 128);

        self.lpc.synthesize_with_params(
            &mut self.out_buffer[..buffer_size],
            &coefs,
            safe_order,
            final_power,
            // Hard safety net against 0.0 or NaN pitch
            noise_adjusted_pitch.max(1.0),
            glottal,
            detune,
            sample_hold_period,
        );

        if preemphasis {
            self.lpc.deemphasis(&mut self.out_buffer[..buffer_size], 0.5);
        }
        
        for sample in self.out_buffer[..buffer_size].iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
            } else {
                *sample = sample.max(-10.0).min(10.0);
            }
        }
    }
}

impl Vst3Plugin for LpcPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"LPCSynthVST3____";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Fx, Vst3SubCategory::Synth];
}
