/// Linear Predictive Coding (LPC) Processor Implementation
/// 
/// This module implements LPC analysis and synthesis using the autocorrelation
/// method with a Rosenberg glottal flow model for voiced excitation.

/// Maximum buffer size supported (4096 samples)
pub const MAX_BUFFER_SIZE: usize = 4096;

/// Default LPC order (LPC-10 for retro sound)
pub const DEFAULT_ORDER: usize = 10;

/// Rosenberg glottal flow table size
const GLOTTAL_TABLE_SIZE: usize = 256;
/// Rosenberg opening duration (as fraction of total period)
const OPENING_DURATION: f32 = 0.5;
/// Rosenberg closing duration (as fraction of total period)
const CLOSING_DURATION: f32 = 0.4;

/// Minimum pitch period in samples (for 20Hz at 44100Hz sample rate)
const MIN_PITCH: usize = 20;
/// Maximum pitch period in samples (for 80Hz)
const MAX_PITCH: usize = 2205;

/// Linear Congruential Generator seed for deterministic unvoiced noise
const LCG_SEED: u32 = 12345;
/// LCG multiplier (numerical recipes)
const LCG_MULTIPLIER: u32 = 1664525;
/// LCG increment
const LCG_INCREMENT: u32 = 1013904223;

/// 10-bit quantization levels (Speak & Spell authentic)
const QUANTIZATION_LEVELS: f32 = 1024.0;

/// LPC Processor state
pub struct LpcProcessor {
    /// Glottal excitation table (Rosenberg glottal FLOW, not derivative)
    glottal_table: Vec<f32>,
    
    /// Window function buffer (Hamming)
    window: Vec<f32>,
    
    /// State buffer for the all-pole filter
    filter_state: Vec<f32>,
    
    /// LCG state for deterministic random generation
    lcg_state: u32,
    
    /// Current position in glottal table for voiced synthesis
    glottal_pos: f32,
    
    /// Voiced excitation ticker
    ticker: i32,
    
    /// Preemphasis state (persists across buffer boundaries)
    preemphasis_state: f32,
    
    /// Deemphasis state (persists across buffer boundaries)
    deemphasis_state: f32,
    
    /// Sample & hold state for retro effect
    held_sample: f32,
}

impl LpcProcessor {
    /// Create a new LPC processor
    /// 
    /// Initializes the processor by:
    /// 1. Pre-computing a Hamming window function
    /// 2. Synthesizing an internal glottal flow table via the Rosenberg model
    pub fn new() -> Self {
        // Generate glottal flow table using Rosenberg model (FLOW, not derivative)
        let glottal_table = Self::generate_glottal_flow_table();
        
        // Pre-compute Hamming window (max buffer size)
        let window = Self::generate_hamming_window(MAX_BUFFER_SIZE);
        
        // Filter state for all-pole filter
        let filter_state = vec![0.0f32; DEFAULT_ORDER];
        
        // Initialize LCG with seed for deterministic output
        let lcg_state = LCG_SEED;
        
        Self {
            glottal_table,
            window,
            filter_state,
            lcg_state,
            glottal_pos: 0.0,
            ticker: 0,
            preemphasis_state: 0.0,
            deemphasis_state: 0.0,
            held_sample: 0.0,
        }
    }
    
    /// Generate the Rosenberg glottal FLOW table (not derivative)
    /// 
    /// The Rosenberg model defines glottal flow g(t) as:
    /// - Opening phase (0 <= t < T_o): A * t²
    /// - Closing phase (T_o <= t < T_o + T_c): A * (T - t)²
    /// - Closed phase: 0
    /// 
    /// This is the actual flow, not its derivative, for cleaner excitation.
    fn generate_glottal_flow_table() -> Vec<f32> {
        let mut table = vec![0.0f32; GLOTTAL_TABLE_SIZE];
        
        let t_open = OPENING_DURATION;
        let t_close = OPENING_DURATION + CLOSING_DURATION;
        
        // Generate glottal FLOW (not derivative)
        for i in 0..GLOTTAL_TABLE_SIZE {
            let t = i as f32 / GLOTTAL_TABLE_SIZE as f32;
            let flow = if t < t_open {
                // Opening phase: A * t²
                // At t = T_o, we want flow = 1.0, so A = 1/T_o²
                let a = 1.0 / (t_open * t_open);
                a * t * t
            } else if t < t_close {
                // Closing phase: A * (T - t)²
                // At t = T_o, flow = 1.0, at t = T_o+T_c, flow = 0
                let a = 1.0 / ((t_close - t_open) * (t_close - t_open));
                let tau = t_close - t;
                a * tau * tau
            } else {
                // Closed phase: 0
                0.0
            };
            
            table[i] = flow;
        }
        
        table
    }
    
    /// Generate Hamming window function
    /// Hamming window: w(n) = 0.54 - 0.46 * cos(2πn / (N-1))
    fn generate_hamming_window(size: usize) -> Vec<f32> {
        let mut window = vec![0.0f32; size];
        let two_pi = 2.0 * std::f32::consts::PI;
        
        for i in 0..size {
            let n = i as f32;
            let n1 = (size - 1) as f32;
            window[i] = 0.54 - 0.46 * (two_pi * n / n1).cos();
        }
        
        window
    }
    
    /// Retrieve a scaled window value for a given index and target buffer size
    pub fn get_window_value(&self, index: usize, buffer_size: usize) -> f32 {
        if index >= buffer_size {
            return 0.0;
        }
        
        // Scale window if buffer size differs from precomputed size
        if buffer_size == self.window.len() {
            self.window[index]
        } else {
            // Compute Hamming window on-the-fly for different sizes
            let two_pi = 2.0 * std::f32::consts::PI;
            let n = index as f32;
            let n1 = (buffer_size - 1) as f32;
            0.54 - 0.46 * (two_pi * n / n1).cos()
        }
    }
    
    /// Quantize sample to 10-bit (Speak & Spell authentic)
    fn quantize(&self, sample: f32) -> f32 {
        // Clamp first
        let clamped = sample.max(-1.0).min(1.0);
        // Quantize to 10-bit
        let quantized = (clamped * QUANTIZATION_LEVELS).round() / QUANTIZATION_LEVELS;
        quantized
    }
    
    /// Analyze an input audio frame to extract LPC coefficients, residual power, and pitch
    /// 
    /// # Returns
    /// A tuple containing `(power, pitch)` where:
    /// * `power` is the root-mean-square of the prediction error (residue)
    /// * `pitch` is the estimated fundamental period in samples (0.0 for unvoiced)
    pub fn analyze(&mut self, input: &[f32], coefs: &mut [f32], order: usize) -> (f32, f32) {
        let frame_size = input.len();
        
        // Guard against empty or too-small input
        if frame_size < order + 2 || input.is_empty() {
            for c in coefs.iter_mut() {
                *c = 0.0;
            }
            return (0.001, 0.0);
        }
        
        // Apply Hamming window function
        let mut windowed_input = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            let w = self.get_window_value(i, frame_size);
            let inp = input[i];
            windowed_input[i] = if inp.is_finite() { inp * w } else { 0.0 };
        }
        
        // Pitch detection: find peak in autocorrelation between MIN_PITCH and MAX_PITCH
        let search_start = MIN_PITCH.min(frame_size / 2);
        let search_end = (MAX_PITCH + 1).min(frame_size / 2);
        let autocorr_len = (order + 1).max(search_end);
        
        // Compute autocorrelation
        let mut autocorr = vec![0.0f32; autocorr_len];
        for lag in 0..autocorr_len {
            let mut sum = 0.0f32;
            for i in 0..(frame_size - lag) {
                sum += windowed_input[i] * windowed_input[i + lag];
            }
            autocorr[lag] = if sum.is_finite() { sum } else { 0.0 };
        }
        
        // Guard against zero or invalid autocorrelation at lag 0
        if !autocorr[0].is_finite() || autocorr[0] <= 0.0 {
            for c in coefs.iter_mut() {
                *c = 0.0;
            }
            return (0.001, 0.0);
        }
        
        // Add a small white noise floor to stabilize the matrix inversion
        autocorr[0] *= 1.001;
        
        // Find pitch peak in autocorrelation with tighter thresholds
        let mut pitch: f32 = 0.0;
        
        if search_start < search_end && search_start < autocorr.len() {
            // Tighter threshold: require at least 40% of zero-lag for voiced
            let threshold = autocorr[0] * 0.4;
            let mut max_corr = threshold;
            let mut best_lag = 0;
            
            // Also check for local maxima to avoid detecting harmonics
            for lag in search_start..search_end {
                if lag < autocorr.len() && autocorr[lag] > max_corr {
                    // Check if this is a local maximum (not a harmonic of a lower peak)
                    let is_local_max = if lag > search_start && lag < search_end - 1 {
                        autocorr[lag] > autocorr[lag - 1] && autocorr[lag] > autocorr[lag + 1]
                    } else {
                        true
                    };
                    
                    if is_local_max {
                        max_corr = autocorr[lag];
                        best_lag = lag;
                    }
                }
            }
            
            if best_lag > 0 {
                pitch = best_lag as f32;
            }
        }
        
        // Check if voiced (pitch > 0) or unvoiced
        let is_voiced = pitch > 0.0;
        
        // Calculate LPC coefficients using autocorrelation method (Levinson-Durbin algorithm)
        let mut lpc = vec![0.0f32; order];
        
        // Levinson-Durbin recursion
        let mut err = autocorr[0];
        let mut temp_lpc = vec![0.0f32; order];
        
        for i in 0..order {
            // Calculate reflection coefficient
            let mut k = autocorr[i + 1];
            
            for j in 0..i {
                k -= temp_lpc[j] * autocorr[i - j];
            }
            
            // Guard against division by zero
            if err.abs() > 1e-10 {
                k /= err;
            } else {
                break;
            }
            
            // Guard against unstable reflection coefficient
            if !k.is_finite() || k.abs() > 0.99 {
                k = k.signum() * 0.99;
            }
            
            // Clone temp_lpc before updating in-place!
            let prev_lpc = temp_lpc.clone();
            
            // Update LPC coefficients
            temp_lpc[i] = k;
            
            for j in 0..i {
                temp_lpc[j] = prev_lpc[j] - k * prev_lpc[i - 1 - j];
            }
            
            err *= 1.0 - k * k;
            
            // Check for singular matrix
            if !err.is_finite() || err <= 0.0 {
                break;
            }
        }
        
        // Copy to output
        lpc.copy_from_slice(&temp_lpc);
        
        // Apply bandwidth expansion with HIGHER gamma (less dampening to preserve formants)
        // gamma = 0.998 allows more formants through
        let gamma = 0.998f32;
        let mut multiplier = gamma;
        
        for coef in lpc.iter_mut() {
            if !coef.is_finite() {
                *coef = 0.0;
            } else {
                // Apply bandwidth expansion
                *coef *= multiplier;
            }
            multiplier *= gamma;
        }
        
        // Copy coefficients to output
        let coefs_len = coefs.len().min(order);
        coefs[..coefs_len].copy_from_slice(&lpc[..coefs_len]);
        for i in coefs_len..coefs.len() {
            coefs[i] = 0.0;
        }
        
        // Use Levinson-Durbin's 'err' as the total residual energy
        let power = if err > 0.0 {
            (err / frame_size as f32).sqrt()
        } else {
            0.001
        };
        
        // Validate power
        if !power.is_finite() || power <= 0.0 {
            return (0.001, if is_voiced { pitch } else { 0.0 });
        }
        
        // Return power and pitch (0.0 for unvoiced)
        (power, if is_voiced { pitch } else { 0.0 })
    }
    
    /// Synthesize with extended parameters including detune and sample hold period
    /// 
    /// This method combines synthesis with detune (second oscillator) and configurable
    /// sample & hold period for rate control.
    pub fn synthesize_with_params(
        &mut self,
        output: &mut [f32],
        coefs: &[f32],
        order: usize,
        power: f32,
        pitch: f32,
        use_glottal: bool,
        detune_cents: f32,
        sample_hold_period: usize,
    ) {
        let frame_size = output.len();
        
        // Guard against empty output
        if frame_size == 0 || order == 0 {
            return;
        }
        
        // Validate inputs
        let valid_power = if power.is_finite() && power > 0.0 { power } else { 0.001 };
        let valid_pitch = if pitch.is_finite() && pitch > 0.0 { pitch } else { 0.0 };
        
        // Calculate detune frequency multiplier
        let detune_mult = if detune_cents > 0.0 {
            2.0f32.powf(detune_cents / 1200.0)
        } else {
            1.0f32
        };
        
        // Ensure filter state is large enough
        if self.filter_state.len() < order {
            self.filter_state = vec![0.0f32; order];
        }
        
        // Clear output
        for sample in output.iter_mut() {
            *sample = 0.0;
        }
        
        // Calculate pitch gain to compensate for sparse impulse train energy
        let pitch_gain = if valid_pitch > 0.0 { valid_pitch.sqrt() } else { 1.0 };
        
        // For detune, we need two oscillators
        let detune_pitch = if valid_pitch > 0.0 && detune_cents > 0.0 {
            Some(valid_pitch / detune_mult)
        } else {
            None
        };
        
        // Generate excitation and apply all-pole filter
        for i in 0..frame_size {
            let excitation = if valid_pitch <= 0.0 {
                // Unvoiced: deterministic pseudo-random noise
                self.deterministic_rand_2() * valid_power
            } else {
                // Voiced excitation - main oscillator using Rosenberg GLOTTAL FLOW
                if self.ticker <= 0 {
                    // Reset at pitch period
                    self.ticker = valid_pitch as i32;
                    
                    if use_glottal {
                        // Use Rosenberg glottal FLOW (not derivative) - no decay
                        self.glottal_pos = 0.0;
                        let table_idx = (self.glottal_pos * GLOTTAL_TABLE_SIZE as f32) as usize;
                        let glottal_sample = if table_idx < GLOTTAL_TABLE_SIZE {
                            self.glottal_table[table_idx]
                        } else {
                            0.0
                        };
                        // Scale by power and pitch_gain - NO DECAY
                        glottal_sample * valid_power * pitch_gain
                    } else {
                        // Simple impulse with pitch_gain
                        valid_power * pitch_gain
                    }
                } else {
                    // Between pulses - zeros for clean excitation
                    // NO glottal pulse continuation - just silence
                    0.0
                }
            };
            
            // Decrement ticker
            self.ticker -= 1;
            
            // All-pole filter: y[n] = excitation[n] + sum(a[i] * y[n-i])
            let mut new_sample = excitation;
            
            // Add the filter contribution from past outputs
            for j in 0..order {
                if j < self.filter_state.len() {
                    let coef = coefs[j];
                    if coef.is_finite() {
                        new_sample += coef * self.filter_state[j];
                    }
                }
            }
            
            // Guard against NaN/Inf in output
            if !new_sample.is_finite() {
                new_sample = 0.0;
            }
            
            // Clamp to prevent explosion
            new_sample = new_sample.max(-100.0).min(100.0);
            
            // Shift filter state and insert new sample at the front
            for j in (1..self.filter_state.len()).rev() {
                self.filter_state[j] = self.filter_state[j - 1];
            }
            if !self.filter_state.is_empty() {
                self.filter_state[0] = new_sample;
            }
            
            output[i] = new_sample;
        }
        
        // Apply detune: add second oscillator if detune > 0
        if let Some(detuned_pitch) = detune_pitch {
            if detuned_pitch > 0.0 && detuned_pitch < frame_size as f32 {
                let mut detune_ticker: i32 = 0;
                let detune_pitch_gain = detuned_pitch.sqrt();
                
                for i in 0..frame_size {
                    let detune_excitation = if detune_ticker <= 0 {
                        detune_ticker = detuned_pitch as i32;
                        if use_glottal {
                            let table_idx = 0; // Start of glottal flow
                            let glottal_sample = if table_idx < GLOTTAL_TABLE_SIZE {
                                self.glottal_table[table_idx]
                            } else {
                                0.0
                            };
                            glottal_sample * valid_power * detune_pitch_gain * 0.5
                        } else {
                            valid_power * detune_pitch_gain * 0.5
                        }
                    } else {
                        0.0
                    };
                    
                    detune_ticker -= 1;
                    
                    // Mix detune with original
                    output[i] += detune_excitation;
                }
            }
        }
        
        // Apply sample & hold with 10-bit quantization for authenticity
        // This creates the characteristic Speak & Spell "crunchy" sound
        if sample_hold_period > 1 {
            let hold_period = sample_hold_period.max(1);
            let mut sample_counter = 0;
            
            for sample in output.iter_mut() {
                if sample_counter == 0 {
                    self.held_sample = *sample;
                }
                *sample = self.held_sample;
                
                // Apply 10-bit quantization for authentic retro sound
                *sample = self.quantize(*sample);
                
                sample_counter = (sample_counter + 1) % hold_period;
            }
        } else {
            // Even without S&H, apply quantization for authenticity
            for sample in output.iter_mut() {
                *sample = self.quantize(*sample);
            }
        }
        
        // Final safety clamp
        for sample in output.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
            } else {
                *sample = sample.max(-10.0).min(10.0);
            }
        }
    }
    
    /// Synthesize an audio frame from LPC coefficients, power, and pitch
    pub fn synthesize(
        &mut self,
        output: &mut [f32],
        coefs: &[f32],
        order: usize,
        power: f32,
        pitch: f32,
        use_alt: bool,
    ) {
        // Delegate to synthesize_with_params with default settings
        self.synthesize_with_params(
            output,
            coefs,
            order,
            power,
            pitch,
            use_alt,
            0.0,  // No detune
            5,    // Default hold period for 8kHz effect
        );
    }
    
    /// Deterministic pseudo-random number generator (Linear Congruential Generator)
    /// Returns value in range [-1, 1]
    fn deterministic_rand_2(&mut self) -> f32 {
        self.lcg_state = LCG_MULTIPLIER.wrapping_mul(self.lcg_state).wrapping_add(LCG_INCREMENT);
        // Convert to float in range [0, 1]
        let value = (self.lcg_state >> 16) as f32 / 65536.0;
        // Map to [-1, 1]
        value * 2.0 - 1.0
    }
    
    /// Apply preemphasis filter to signal (stateful, persists across buffers)
    /// y[n] = x[n] - alpha * x[n-1]
    pub fn preemphasis(&mut self, signal: &mut [f32], alpha: f32) {
        for i in 0..signal.len() {
            let current = signal[i];
            signal[i] = current - alpha * self.preemphasis_state;
            self.preemphasis_state = current;
        }
    }
    
    /// Apply deemphasis filter to signal (stateful, persists across buffers)
    /// y[n] = x[n] + alpha * y[n-1]
    pub fn deemphasis(&mut self, signal: &mut [f32], alpha: f32) {
        for i in 0..signal.len() {
            let current = signal[i];
            let new_val = current + alpha * self.deemphasis_state;
            self.deemphasis_state = new_val;
            signal[i] = new_val;
        }
    }
    
    /// Reset filter states (call when plugin resets)
    pub fn reset_states(&mut self) {
        self.preemphasis_state = 0.0;
        self.deemphasis_state = 0.0;
        for s in self.filter_state.iter_mut() {
            *s = 0.0;
        }
        // Reset synthesis state to ensure immediate silence
        self.glottal_pos = 0.0;
        self.ticker = 0;
        self.lcg_state = LCG_SEED;
        self.held_sample = 0.0;
    }
}

impl Default for LpcProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_glottal_flow_table_generation() {
        let processor = LpcProcessor::new();
        assert_eq!(processor.glottal_table.len(), GLOTTAL_TABLE_SIZE);
        
        // Check that max value is 1.0 (flow, not derivative)
        let max_val = processor.glottal_table.iter().fold(0.0f32, |a, b| a.max(*b));
        assert!((max_val - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_hamming_window_function() {
        let processor = LpcProcessor::new();
        
        // Check window values are in [0.08, 1.0] for Hamming
        for i in 0..processor.window.len() {
            assert!(processor.window[i] >= 0.08 && processor.window[i] <= 1.0);
        }
    }
    
    #[test]
    fn test_deterministic_rand() {
        let mut processor = LpcProcessor::new();
        
        // Get first value
        let val1 = processor.deterministic_rand_2();
        
        // Reset to same seed
        processor.lcg_state = LCG_SEED;
        
        // Get value again should be the same
        let val2 = processor.deterministic_rand_2();
        
        assert_eq!(val1, val2);
    }
    
    #[test]
    fn test_quantization() {
        let processor = LpcProcessor::new();
        
        // Test quantization to 10-bit levels
        let test_val = 0.5;
        let quantized = processor.quantize(test_val);
        
        // Should be snapped to a quantization level
        let expected = (0.5 * QUANTIZATION_LEVELS).round() / QUANTIZATION_LEVELS;
        assert!((quantized - expected).abs() < 0.001);
    }
    
    #[test]
    fn test_preemphasis_deemphasis() {
        let mut processor = LpcProcessor::new();
        let mut signal = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let original = signal.clone();
        
        // Test with 0.0 alpha (identity filter)
        processor.preemphasis(&mut signal, 0.0);
        processor.deemphasis(&mut signal, 0.0);
        
        // Allow small numerical error
        for i in 0..signal.len() {
            assert!((signal[i] - original[i]).abs() < 0.01);
        }
    }
}
