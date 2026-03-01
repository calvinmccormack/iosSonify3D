import Foundation
import AVFoundation
import Combine

/// Pure FM synthesis engine with up to 4 target voices + scan drone.
///
/// Design:
/// - Each voice's timbre (C:M ratio) is fixed per SLOT, not per class.
///   Slot 1–4 have maximally distinct timbres.
/// - Y centroid → carrier frequency (consistent across all slots).
/// - Z depth   → amplitude (near = loud, far = quiet).
/// - Coverage  → modulation index β (tall column slice = bright/rich,
///   thin slice = pure). This conveys object height per column.
/// - Pan       → stereo position from sweep X.
/// - A low triangle-wave drone pans with the sweep to indicate scan
///   position, start, and end.
final class SpectralAudioEngine: ObservableObject {

    @Published var outputGainDB: Float = 6.0

    private let sampleRate: Double
    private let engine = AVAudioEngine()
    private var srcNode: AVAudioSourceNode!

    // ─── Slot timbres: maximally distinct C:M ratios ───
    // ALL ratios >= 1.0 so the carrier always determines perceived pitch.
    // The modulator colors the spectrum ABOVE the carrier fundamental.
    // Slot 0: 1:1   → all harmonics (warm, brass-like)
    // Slot 1: 1:2   → odd harmonics (hollow, clarinet-like)
    // Slot 2: 1:3   → every-3rd harmonic (bell-like, bright)
    // Slot 3: 1:1.5 → perfect-fifth harmonics (nasal, distinct)
    static let slotCMRatios: [Float] = [1.0, 2.0, 3.0, 1.5]

    // ─── FM Voices ───
    struct FMVoice {
        var slot: Int = -1       // 0-3
        var classId: Int = -1    // which COCO class is assigned (for display only)
        var active: Bool = false
        var level: Float = 0
        var targetLevel: Float = 0

        var cmRatio: Float = 1.0
        var carrierPhase: Float = 0
        var modPhase: Float = 0

        // Spatial params (smoothed)
        var carrierFreq: Float = 300
        var targetCarrierFreq: Float = 300
        var modIndex: Float = 1.0      // driven by COVERAGE (object height)
        var targetModIndex: Float = 1.0
        var amplitude: Float = 0       // driven by DEPTH (nearness)
        var targetAmplitude: Float = 0

        mutating func reset() {
            slot = -1; classId = -1; active = false
            level = 0; targetLevel = 0
            carrierPhase = 0; modPhase = 0
            carrierFreq = 300; targetCarrierFreq = 300
            modIndex = 1.0; targetModIndex = 1.0
            amplitude = 0; targetAmplitude = 0
        }
    }

    private var voices: [FMVoice] = (0..<4).map { _ in FMVoice() }
    private let levelRamp: Float = 0.92
    private let paramRamp: Float = 0.82

    // ─── Consistent Y → pitch mapping ───
    // centroid 0 (bottom) → 120 Hz, centroid 1 (top) → 800 Hz
    // Log scale for perceptual linearity
    private let pitchMinHz: Float = 120
    private let pitchMaxHz: Float = 800

    // ─── Coverage → modulation index mapping ───
    // coverage 0 → β = 0.3 (pure tone), coverage 1 → β = 6.0 (very bright)
    private let betaMin: Float = 0.3
    private let betaMax: Float = 6.0

    // ─── Scan drone ───
    private var dronePhase: Float = 0
    private let droneFreq: Float = 90          // Hz — low but audible on bone conduction
    private let droneAmplitude: Float = 0.06   // subtle
    private var droneActive: Bool = false
    private var dronePan: Float = 0            // follows sweep
    private var droneTargetPan: Float = 0
    private var droneLevel: Float = 0          // for fade in/out
    private var droneTargetLevel: Float = 0

    // ─── Smoothed output pan per voice ───
    private var voicePan: [Float] = [0, 0, 0, 0]
    private var voiceTargetPan: [Float] = [0, 0, 0, 0]

    init() {
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playAndRecord, mode: .default,
                                 options: [.defaultToSpeaker, .allowBluetooth, .mixWithOthers])
        try? session.setActive(true)
        self.sampleRate = session.sampleRate

        let fmt = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        srcNode = AVAudioSourceNode(format: fmt) { [weak self] _, _, frameCount, buf -> OSStatus in
            guard let self else { return noErr }
            let abl = UnsafeMutableAudioBufferListPointer(buf)
            guard let left = abl[0].mData?.assumingMemoryBound(to: Float.self),
                  let right = abl[1].mData?.assumingMemoryBound(to: Float.self) else { return noErr }
            self.render(left: left, right: right, count: Int(frameCount))
            return noErr
        }
        engine.attach(srcNode)
        engine.connect(srcNode, to: engine.mainMixerNode, format: fmt)
    }

    func start() { if !engine.isRunning { try? engine.start() } }
    func stop() { engine.stop() }

    // MARK: - Scan lifecycle

    func beginScan() {
        droneActive = true
        droneTargetLevel = 1.0
        print("[Audio] beginScan — drone on, engine running: \(engine.isRunning)")
        // Ensure engine is running (TTS may have interrupted it)
        if !engine.isRunning { try? engine.start() }
    }

    func endScan() {
        droneTargetLevel = 0
        for i in 0..<4 {
            voices[i].targetLevel = 0
            voices[i].targetAmplitude = 0
        }
    }

    // MARK: - Target routing

    /// Called per sweep column when target object is detected.
    /// `slot`: 0-3, which target slot this class is assigned to.
    func setTarget(slot: Int, centroid: Float, coverage: Float, depth: Float, pan: Float) {
        guard slot >= 0 && slot < 4 else { return }

        // Y centroid → carrier frequency (log scale)
        let logMin = log2(pitchMinHz)
        let logMax = log2(pitchMaxHz)
        let freq = powf(2.0, logMin + centroid * (logMax - logMin))
        voices[slot].targetCarrierFreq = freq

        // Coverage → modulation index (object height → brightness)
        let beta = betaMin + min(1.0, coverage * 2.0) * (betaMax - betaMin)
        voices[slot].targetModIndex = beta

        // Depth → amplitude (near = loud, far = quiet)
        let nearness = max(0, min(1, 1.0 - depth))
        voices[slot].targetAmplitude = 0.2 + nearness * 0.8

        voices[slot].targetLevel = 1.0
        voiceTargetPan[slot] = pan

        // Ensure voice is configured
        if !voices[slot].active {
            voices[slot].active = true
            voices[slot].cmRatio = Self.slotCMRatios[slot]
            voices[slot].slot = slot
            print("[Audio] Activated slot \(slot) cm=\(voices[slot].cmRatio) fc=\(Int(freq))Hz β=\(String(format:"%.1f",beta)) amp=\(String(format:"%.2f",voices[slot].targetAmplitude))")
        }

        // Update drone pan to follow sweep
        droneTargetPan = pan
    }

    /// Called for columns with no target — decay voices.
    func clearColumn(pan: Float) {
        droneTargetPan = pan
        for i in 0..<4 {
            voices[i].targetLevel *= 0.90
            voices[i].targetAmplitude *= 0.85
        }
    }

    /// Assign a class to a slot (called when user picks target).
    func assignSlot(_ slot: Int, classId: Int) {
        guard slot >= 0 && slot < 4 else { return }
        voices[slot].slot = slot
        voices[slot].classId = classId
        voices[slot].cmRatio = Self.slotCMRatios[slot]
        voices[slot].active = true
    }

    /// Deactivate all voices.
    func deactivateAll() {
        for i in 0..<4 { voices[i].reset() }
        droneActive = false; droneLevel = 0; droneTargetLevel = 0
    }

    // MARK: - Render

    private func render(left: UnsafeMutablePointer<Float>,
                         right: UnsafeMutablePointer<Float>, count: Int) {
        let sr = Float(sampleRate)
        let twoPi = Float.pi * 2.0
        let gain = powf(10.0, outputGainDB / 20.0)

        // Per-block param smoothing
        for i in 0..<4 {
            voices[i].level       = levelRamp * voices[i].level       + (1 - levelRamp) * voices[i].targetLevel
            voices[i].carrierFreq = paramRamp * voices[i].carrierFreq + (1 - paramRamp) * voices[i].targetCarrierFreq
            voices[i].modIndex    = paramRamp * voices[i].modIndex    + (1 - paramRamp) * voices[i].targetModIndex
            voices[i].amplitude   = paramRamp * voices[i].amplitude   + (1 - paramRamp) * voices[i].targetAmplitude
            voicePan[i] += (voiceTargetPan[i] - voicePan[i]) * 0.15

            if voices[i].active && voices[i].targetLevel < 0.001 && voices[i].level < 0.001 {
                voices[i].reset()
            }
        }

        // Drone smoothing
        droneLevel += (droneTargetLevel - droneLevel) * 0.05
        dronePan += (droneTargetPan - dronePan) * 0.15
        if droneLevel < 0.001 && droneTargetLevel < 0.001 { droneActive = false }

        for s in 0..<count {
            var sampleL: Float = 0
            var sampleR: Float = 0

            // FM voices
            for i in 0..<4 {
                guard voices[i].active && voices[i].level > 0.003 else { continue }

                let fc = voices[i].carrierFreq
                let fmFreq = fc * voices[i].cmRatio
                let beta = voices[i].modIndex

                let modSig = sinf(voices[i].modPhase)
                let out = sinf(voices[i].carrierPhase + beta * modSig)
                let amp = out * voices[i].level * voices[i].amplitude * 0.35

                // Per-voice pan
                let theta = (-voicePan[i] + 1) * Float.pi * 0.25
                sampleL += amp * sin(theta)
                sampleR += amp * cos(theta)

                voices[i].carrierPhase += twoPi * fc / sr
                voices[i].modPhase += twoPi * fmFreq / sr
                if voices[i].carrierPhase > twoPi { voices[i].carrierPhase -= twoPi }
                if voices[i].modPhase > twoPi { voices[i].modPhase -= twoPi }
            }

            // Scan drone (triangle wave)
            if droneActive || droneLevel > 0.001 {
                // Triangle wave: 4 * |phase/2π - 0.5| - 1
                let normPhase = dronePhase / twoPi
                let tri = 4.0 * abs(normPhase - 0.5) - 1.0
                let droneSig = tri * droneAmplitude * droneLevel

                let dTheta = (-dronePan + 1) * Float.pi * 0.25
                sampleL += droneSig * sin(dTheta)
                sampleR += droneSig * cos(dTheta)

                dronePhase += twoPi * droneFreq / sr
                if dronePhase > twoPi { dronePhase -= twoPi }
            }

            // Gain + soft clip
            sampleL *= gain; sampleR *= gain
            sampleL = max(-0.95, min(0.95, sampleL))
            sampleR = max(-0.95, min(0.95, sampleR))

            left[s] = sampleL
            right[s] = sampleR
        }

        // Debug
        struct Dbg { static var t: Double = 0 }
        let now = CACurrentMediaTime()
        if now - Dbg.t > 0.5 {
            Dbg.t = now
            let anyActive = voices.contains { $0.active && $0.level > 0.01 }
            if anyActive || droneLevel > 0.01 {
                var parts: [String] = []
                for i in 0..<4 where voices[i].active {
                    parts.append("s\(i):fc=\(Int(voices[i].carrierFreq)) β=\(String(format:"%.1f",voices[i].modIndex)) amp=\(String(format:"%.2f",voices[i].amplitude)) lvl=\(String(format:"%.2f",voices[i].level))")
                }
                if droneLevel > 0.01 { parts.append("drone=\(String(format:"%.2f",droneLevel))") }
                print("[FM] \(parts.joined(separator: " | "))")
            }
        }
    }

    // MARK: - Stubs for API compat
    func configureBands(fMin: Double, fMax: Double) {}
    func updateEnvelope(_ env: [Float]) {}
    func updateDistance(_ z: Float) {}
    func triggerEdge(_ e: Float) {}
}
