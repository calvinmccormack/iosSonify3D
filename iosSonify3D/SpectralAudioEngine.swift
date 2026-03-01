import Foundation
import AVFoundation
import Combine

/// Pure FM synthesis engine — no background noise layer.
/// Two FM voices for up to 2 simultaneous object classes.
/// All spatial parameters driven by the sweep snapshot.
final class SpectralAudioEngine: ObservableObject {

    @Published var pan: Float = 0
    @Published var outputGainDB: Float = 6.0

    private let sampleRate: Double
    private let engine = AVAudioEngine()
    private var srcNode: AVAudioSourceNode!

    // ─── FM Voices ───
    struct FMVoice {
        var classId: Int = -1
        var active: Bool = false
        var level: Float = 0
        var targetLevel: Float = 0

        var cmRatio: Float = 1.0
        var carrierPhase: Float = 0
        var modPhase: Float = 0

        var carrierFreq: Float = 220
        var targetCarrierFreq: Float = 220
        var modIndex: Float = 2.0
        var targetModIndex: Float = 2.0
        var ampEnvelope: Float = 0
        var targetAmpEnvelope: Float = 0

        mutating func reset() {
            classId = -1; active = false; level = 0; targetLevel = 0
            carrierPhase = 0; modPhase = 0
            carrierFreq = 220; targetCarrierFreq = 220
            modIndex = 2.0; targetModIndex = 2.0
            ampEnvelope = 0; targetAmpEnvelope = 0
        }
    }

    private var fm1 = FMVoice()
    private var fm2 = FMVoice()
    private let levelRamp: Float = 0.93
    private let paramRamp: Float = 0.85

    // Smoothed pan
    private var currentPan: Float = 0
    private var targetPan: Float = 0

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
            self.renderFM(left: left, right: right, count: Int(frameCount))
            return noErr
        }
        engine.attach(srcNode)
        engine.connect(srcNode, to: engine.mainMixerNode, format: fmt)
    }

    func start() { if !engine.isRunning { try? engine.start() } }
    func stop() { engine.stop() }

    // MARK: - Target routing

    /// Called by sweep for each column that has a detected object.
    func setTarget(classId: Int, centroid: Float, coverage: Float, depth: Float, pan: Float) {
        targetPan = pan

        if classId == fm1.classId || (fm1.classId == -1 && classId != fm2.classId) {
            configureFMVoice(&fm1, classId: classId, centroid: centroid,
                             coverage: coverage, depth: depth)
            fm1.targetLevel = 1.0
        } else if classId == fm2.classId || fm2.classId == -1 {
            configureFMVoice(&fm2, classId: classId, centroid: centroid,
                             coverage: coverage, depth: depth)
            fm2.targetLevel = 1.0
        } else {
            configureFMVoice(&fm1, classId: classId, centroid: centroid,
                             coverage: coverage, depth: depth)
            fm1.targetLevel = 1.0
        }
    }

    /// Called for columns with no target detection — decay FM.
    func clearTarget(pan: Float) {
        targetPan = pan
        fm1.targetLevel *= 0.92
        fm2.targetLevel *= 0.92
        fm1.targetAmpEnvelope *= 0.88
        fm2.targetAmpEnvelope *= 0.88
    }

    /// Called when user clears all targets.
    func deactivateAll() {
        fm1.reset(); fm2.reset()
    }

    /// Called when scan finishes — let FM ring out then silence.
    func endScan() {
        fm1.targetLevel = 0
        fm2.targetLevel = 0
        fm1.targetAmpEnvelope = 0
        fm2.targetAmpEnvelope = 0
    }

    // MARK: - FM Configuration

    private static func cmRatioForClass(_ classId: Int) -> Float {
        switch classId {
        case 0:       return 1.0     // person — warm brass
        case 1...8:   return 2.01    // vehicles — bright metallic
        case 9...13:  return 1.5     // outdoor — mellow reed
        case 14...23: return 3.0     // animals — bell
        case 24...28: return 2.5     // accessories — hollow/woody
        case 29...38: return 1.414   // sports — bright string
        case 39...45: return 3.5     // kitchen — chime
        case 46...55: return 2.0     // food — clarinet
        case 56...61: return 1.33    // furniture — organ
        case 62...67: return 3.01    // electronics — shimmer
        case 68...72: return 2.5     // appliances — hollow
        case 73...79: return 4.0     // misc — complex bell
        default:      return 1.0
        }
    }

    private static func baseFreqForClass(_ classId: Int) -> Float {
        let phi: Float = 1.618034
        let hash = fmod(Float(classId) * phi, 1.0)
        return powf(2.0, log2(110.0) + hash * (log2(440.0) - log2(110.0)))
    }

    private func configureFMVoice(_ voice: inout FMVoice, classId: Int,
                                   centroid: Float, coverage: Float, depth: Float) {
        if voice.classId != classId {
            voice.classId = classId
            voice.active = true
            voice.cmRatio = Self.cmRatioForClass(classId)
        }
        if !voice.active { voice.active = true }

        let baseFreq = Self.baseFreqForClass(classId)
        let centroidShift = powf(2.0, (centroid - 0.5) * 2.0)
        voice.targetCarrierFreq = baseFreq * centroidShift

        let nearness = 1.0 - depth
        voice.targetModIndex = 0.5 + nearness * 5.0

        voice.targetAmpEnvelope = min(1.0, coverage * 2.0)
    }

    // MARK: - Render

    private func renderFM(left: UnsafeMutablePointer<Float>,
                           right: UnsafeMutablePointer<Float>, count: Int) {
        let sr = Float(sampleRate)
        let twoPi = Float.pi * 2.0
        let gain = powf(10.0, outputGainDB / 20.0)

        // Per-block param smoothing (cheaper than per-sample)
        fm1.level = levelRamp * fm1.level + (1 - levelRamp) * fm1.targetLevel
        fm2.level = levelRamp * fm2.level + (1 - levelRamp) * fm2.targetLevel
        fm1.carrierFreq = paramRamp * fm1.carrierFreq + (1 - paramRamp) * fm1.targetCarrierFreq
        fm1.modIndex    = paramRamp * fm1.modIndex    + (1 - paramRamp) * fm1.targetModIndex
        fm1.ampEnvelope = paramRamp * fm1.ampEnvelope + (1 - paramRamp) * fm1.targetAmpEnvelope
        fm2.carrierFreq = paramRamp * fm2.carrierFreq + (1 - paramRamp) * fm2.targetCarrierFreq
        fm2.modIndex    = paramRamp * fm2.modIndex    + (1 - paramRamp) * fm2.targetModIndex
        fm2.ampEnvelope = paramRamp * fm2.ampEnvelope + (1 - paramRamp) * fm2.targetAmpEnvelope

        if fm1.active && fm1.targetLevel < 0.001 && fm1.level < 0.001 { fm1.reset() }
        if fm2.active && fm2.targetLevel < 0.001 && fm2.level < 0.001 { fm2.reset() }

        // Smooth pan per-block
        currentPan += (targetPan - currentPan) * 0.15

        let theta = (-currentPan + 1) * Float.pi * 0.25
        let gL = sin(theta), gR = cos(theta)

        let hasAny = (fm1.active && fm1.level > 0.003) || (fm2.active && fm2.level > 0.003)

        for i in 0..<count {
            var sample: Float = 0

            if hasAny {
                if fm1.active && fm1.level > 0.003 && fm1.classId >= 0 {
                    let fc = fm1.carrierFreq
                    let fmFreq = fc * fm1.cmRatio
                    let modSig = sinf(fm1.modPhase)
                    let out = sinf(fm1.carrierPhase + fm1.modIndex * modSig)
                    sample += out * fm1.level * fm1.ampEnvelope * 0.4

                    fm1.carrierPhase += twoPi * fc / sr
                    fm1.modPhase += twoPi * fmFreq / sr
                    if fm1.carrierPhase > twoPi { fm1.carrierPhase -= twoPi }
                    if fm1.modPhase > twoPi { fm1.modPhase -= twoPi }
                }

                if fm2.active && fm2.level > 0.003 && fm2.classId >= 0 {
                    let fc = fm2.carrierFreq
                    let fmFreq = fc * fm2.cmRatio
                    let modSig = sinf(fm2.modPhase)
                    let out = sinf(fm2.carrierPhase + fm2.modIndex * modSig)
                    sample += out * fm2.level * fm2.ampEnvelope * 0.4

                    fm2.carrierPhase += twoPi * fc / sr
                    fm2.modPhase += twoPi * fmFreq / sr
                    if fm2.carrierPhase > twoPi { fm2.carrierPhase -= twoPi }
                    if fm2.modPhase > twoPi { fm2.modPhase -= twoPi }
                }
            }

            // Apply gain + soft clip
            sample *= gain
            if sample > 0.95 { sample = 0.95 }
            else if sample < -0.95 { sample = -0.95 }

            left[i] = sample * gL
            right[i] = sample * gR
        }

        // Debug logging
        struct FMDbg { static var lastLog: Double = 0 }
        let now = CACurrentMediaTime()
        if now - FMDbg.lastLog > 2.0 && hasAny {
            FMDbg.lastLog = now
            if fm1.active {
                print("[FM] v1: \(cocoName(fm1.classId)) fc=\(Int(fm1.carrierFreq))Hz cm=\(String(format:"%.2f",fm1.cmRatio)) β=\(String(format:"%.1f",fm1.modIndex)) amp=\(String(format:"%.2f",fm1.ampEnvelope)) lvl=\(String(format:"%.2f",fm1.level))")
            }
            if fm2.active {
                print("[FM] v2: \(cocoName(fm2.classId)) fc=\(Int(fm2.carrierFreq))Hz cm=\(String(format:"%.2f",fm2.cmRatio)) β=\(String(format:"%.1f",fm2.modIndex)) amp=\(String(format:"%.2f",fm2.ampEnvelope)) lvl=\(String(format:"%.2f",fm2.level))")
            }
        }
    }

    private func cocoName(_ id: Int) -> String {
        let n = ["person","bicycle","car","motorcycle","airplane","bus","train",
                 "truck","boat","traffic light","fire hydrant","stop sign",
                 "parking meter","bench","bird","cat","dog","horse","sheep","cow",
                 "elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
                 "tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
                 "baseball bat","baseball glove","skateboard","surfboard",
                 "tennis racket","bottle","wine glass","cup","fork","knife","spoon",
                 "bowl","banana","apple","sandwich","orange","broccoli","carrot",
                 "hot dog","pizza","donut","cake","chair","couch","potted plant",
                 "bed","dining table","toilet","tv","laptop","mouse","remote",
                 "keyboard","cell phone","microwave","oven","toaster","sink",
                 "refrigerator","book","clock","vase","scissors","teddy bear",
                 "hair drier","toothbrush"]
        return id >= 0 && id < n.count ? n[id] : "?\(id)"
    }

    // MARK: - Stubs for API compat (no-ops, kept so callers compile)
    func configureBands(fMin: Double, fMax: Double) {}
    func updateEnvelope(_ env: [Float]) {}
    func updateDistance(_ z: Float) {}
    func triggerEdge(_ e: Float) {}
}
