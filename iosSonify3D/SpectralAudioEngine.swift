import Foundation
import AVFoundation
import Accelerate
import Combine

final class SpectralAudioEngine: ObservableObject {

    @Published var pan: Float = 0
    @Published var enableEdgeClicks: Bool = false
    @Published var outputGainDB: Float = 6.0

    private let sampleRate: Double
    private let blockSize: Int = 512
    private let hop: Int = 128

    private var agcGain: Float = 1.0
    private let agcAlpha: Float = 0.95
    private let agcTargetRMS: Float = 0.20

    private let engine = AVAudioEngine()
    private var srcNode: AVAudioSourceNode!

    private var window: [Float]
    private var invWindowEnergy: Float
    private let forwardDFT: vDSP.DFT<Float>
    private let inverseDFT: vDSP.DFT<Float>

    private var timeBlock = [Float]()
    private var winTime   = [Float]()
    private var freqReal  = [Float]()
    private var freqImag  = [Float]()
    private var ifftReal  = [Float]()
    private var zeroImag  = [Float]()
    private var scratchImagTime = [Float]()
    private var targetBoostBands = [Float](repeating: 0, count: 40)
    private var targetBoostLin: Float = 1.0
    private let targetBoostSmooth: Float = 0.6

    struct DynamicComb {
        var g: Float; var M: Int; var buf: [Float]; var writeIdx = 0
        init(g: Float, M: Int) {
            self.g = max(0.5, min(0.95, g)); self.M = max(50, min(M, 2000))
            self.buf = [Float](repeating: 0, count: self.M)
        }
        @inline(__always) mutating func tick(_ x: Float, delay: Float) -> Float {
            let d = max(10.0, min(delay, Float(M - 1))); let di = Int(d); let frac = d - Float(di)
            let ri0 = (writeIdx - di + M) % M; let ri1 = (ri0 - 1 + M) % M
            let yOut = buf[ri0] + frac * (buf[ri1] - buf[ri0])
            let out = x + g * yOut; buf[writeIdx] = out; writeIdx = (writeIdx + 1) % M
            if !out.isFinite { for i in 0..<M { buf[i] = 0 }; writeIdx = 0; return 0 }
            return out
        }
        mutating func clear() { for i in 0..<M { buf[i] = 0 }; writeIdx = 0 }
    }

    private var comb1: DynamicComb!
    private var comb2: DynamicComb!
    private var class1Id: Int = -1, class2Id: Int = -1
    private var class1Level: Float = 0, class2Level: Float = 0
    private var class1TargetLevel: Float = 0, class2TargetLevel: Float = 0
    private var class1Active = false, class2Active = false
    private let levelRampAlpha: Float = 0.93

    private var currentCentroid: Float = 0.5, smoothedCentroid: Float = 0.5
    private let centroidSmooth: Float = 0.90

    private var olaL: [Float], olaR: [Float]
    private var olaWrite = 0, olaRead = 0

    private let numBands = 40
    private var bandEdges = [Float]()
    private var binToBand = [Int]()
    private let bandLock = NSLock()
    private var currentGains = [Float](repeating: 1, count: 40)
    private let gainSmooth: Float = 0.85

    private var z01: Float = 0
    private var clickEnv: Float = 0
    private var noiseBuffer = [Float]()
    private var noiseIdx = 0
    private let noisePrimeStep = 7919

    init() {
        let session = AVAudioSession.sharedInstance()

        // Use playAndRecord from the start so mic works without session switching
        try? session.setCategory(.playAndRecord, mode: .default,
                                 options: [.defaultToSpeaker, .allowBluetooth, .mixWithOthers])
        try? session.setActive(true)

        self.sampleRate = session.sampleRate
        let sr = Float(sampleRate)
        comb1 = DynamicComb(g: 0.90, M: Int(sr / 40) + 100)
        comb2 = DynamicComb(g: 0.85, M: Int(sr / 40) + 100)

        window = [Float](repeating: 0, count: blockSize)
        vDSP_hann_window(&window, vDSP_Length(blockSize), Int32(vDSP_HANN_NORM))
        var sum: Float = 0
        vDSP_sve(window, 1, &sum, vDSP_Length(blockSize))
        invWindowEnergy = sum > 0 ? 1.0 / sum : 1.0

        timeBlock = [Float](repeating: 0, count: blockSize)
        winTime   = [Float](repeating: 0, count: blockSize)
        freqReal  = [Float](repeating: 0, count: blockSize)
        freqImag  = [Float](repeating: 0, count: blockSize)
        ifftReal  = [Float](repeating: 0, count: blockSize)
        zeroImag  = [Float](repeating: 0, count: blockSize)
        scratchImagTime = [Float](repeating: 0, count: blockSize)

        forwardDFT = vDSP.DFT(previous: nil, count: blockSize, direction: .forward, transformType: .complexReal, ofType: Float.self)!
        inverseDFT = vDSP.DFT(previous: nil, count: blockSize, direction: .inverse, transformType: .complexReal, ofType: Float.self)!

        noiseBuffer = (0..<65536).map { _ in Float.random(in: -0.35...0.35) }

        let olaLen = 4 * blockSize
        olaL = [Float](repeating: 0, count: olaLen)
        olaR = [Float](repeating: 0, count: olaLen)

        let fmt = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        srcNode = AVAudioSourceNode(format: fmt) { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self else { return noErr }
            let abl = UnsafeMutableAudioBufferListPointer(audioBufferList)
            guard let left = abl[0].mData?.assumingMemoryBound(to: Float.self),
                  let right = abl[1].mData?.assumingMemoryBound(to: Float.self) else { return noErr }
            var remaining = Int(frameCount); var offset = 0
            while remaining > 0 {
                self.synthesizeBlock()
                let chunk = min(remaining, self.hop)
                self.copyFromOLA(to: left.advanced(by: offset), outR: right.advanced(by: offset), count: chunk)
                remaining -= chunk; offset += chunk
            }
            return noErr
        }
        engine.attach(srcNode)
        engine.connect(srcNode, to: engine.mainMixerNode, format: fmt)
    }

    func start() { if !engine.isRunning { try? engine.start() } }
    func stop() { engine.stop() }

    func configureBands(fMin: Double, fMax: Double) {
        let fm = Float(fMin), fM = Float(fMax)
        let newEdges: [Float] = stride(from: 0, through: numBands, by: 1).map { i -> Float in
            let r = Float(i) / Float(numBands); return powf(fm / fM, 1 - r) * fM
        }
        let half = blockSize / 2; let nyq = Float(sampleRate / 2); let binHz = nyq / Float(half)
        var newB2B = [Int](); newB2B.reserveCapacity(half + 1)
        for i in 0...half {
            let f = Float(i) * binHz; var band = 0
            for b in 0..<numBands {
                if f < newEdges[b+1] { band = b; break }
                if b == numBands - 1 { band = numBands - 1 }
            }
            newB2B.append(band)
        }
        bandLock.lock(); bandEdges = newEdges; binToBand = newB2B; bandLock.unlock()
    }

    func updateEnvelope(_ env: [Float]) {
        guard env.count == numBands else { return }
        for i in 0..<numBands { currentGains[i] = gainSmooth * currentGains[i] + (1 - gainSmooth) * env[i] }
    }

    func setTargetBands(_ mask: [Float], classId: Int, boostDB: Float) {
        guard mask.count == numBands else { return }
        targetBoostLin = powf(10.0, boostDB / 20.0)
        for i in 0..<numBands {
            targetBoostBands[i] = targetBoostSmooth * targetBoostBands[i] + (1 - targetBoostSmooth) * mask[i]
        }
        var ws: Float = 0, tw: Float = 0
        for i in 0..<numBands { let w = mask[i]; if w > 0.01 { ws += w * Float(i) / Float(numBands-1); tw += w } }
        currentCentroid = tw > 0.01 ? ws / tw : 0.5

        if classId == class1Id || (class1Id == -1 && classId != class2Id) {
            class1Id = classId; class1TargetLevel = 1.0; class2TargetLevel = 0.0
            if !class1Active { class1Active = true }
        } else if classId == class2Id || class2Id == -1 {
            class2Id = classId; class2TargetLevel = 1.0; class1TargetLevel = 0.0
            if !class2Active { class2Active = true }
        } else {
            class1Id = classId; class1TargetLevel = 1.0; class2TargetLevel = 0.0
            if !class1Active { class1Active = true }
        }
    }

    func clearTarget() {
        // Only decay the spectral boost bands for empty columns.
        // Do NOT touch comb target levels here — the comb should sustain
        // as long as there are active detections somewhere in the scene.
        // Comb levels are only zeroed when activeClasses is cleared.
        for i in 0..<numBands { targetBoostBands[i] *= targetBoostSmooth }
    }

    /// Call this when the user clears all targets (no active classes).
    func deactivateCombs() {
        class1TargetLevel = 0; class2TargetLevel = 0; currentCentroid = 0.5
        for i in 0..<numBands { targetBoostBands[i] = 0 }
    }

    func updateDistance(_ z: Float) { z01 = max(0, min(1, z)) }
    func triggerEdge(_ e: Float) { clickEnv = max(clickEnv, e) }

    private func frequencyRange(for classId: Int) -> (low: Float, high: Float) {
        let phi: Float = 1.618034; let hash = fmod(Float(classId) * phi, 1.0)
        let lc = log2(60.0) + hash * (log2(600.0) - log2(60.0)); let c = powf(2.0, lc)
        return (low: max(40, c / 1.414), high: min(800, c * 1.414))
    }

    private func synthesizeBlock() {
        class1Level = levelRampAlpha * class1Level + (1 - levelRampAlpha) * class1TargetLevel
        class2Level = levelRampAlpha * class2Level + (1 - levelRampAlpha) * class2TargetLevel
        smoothedCentroid = centroidSmooth * smoothedCentroid + (1 - centroidSmooth) * currentCentroid

        if class1Active && class1TargetLevel == 0 && class1Level < 0.001 { class1Active = false; class1Id = -1; comb1.clear() }
        if class2Active && class2TargetLevel == 0 && class2Level < 0.001 { class2Active = false; class2Id = -1; comb2.clear() }

        for i in 0..<blockSize { timeBlock[i] = noiseBuffer[noiseIdx]; noiseIdx = (noiseIdx + noisePrimeStep) % noiseBuffer.count }

        vDSP_vmul(timeBlock, 1, window, 1, &winTime, 1, vDSP_Length(blockSize))
        forwardDFT.transform(inputReal: winTime, inputImaginary: zeroImag, outputReal: &freqReal, outputImaginary: &freqImag)
        applySpectralShaping()
        inverseDFT.transform(inputReal: freqReal, inputImaginary: freqImag, outputReal: &ifftReal, outputImaginary: &scratchImagTime)

        var scale = 1.0 / Float(blockSize); vDSP_vsmul(ifftReal, 1, &scale, &ifftReal, 1, vDSP_Length(blockSize))
        var gain = invWindowEnergy * 4; vDSP_vsmul(ifftReal, 1, &gain, &ifftReal, 1, vDSP_Length(blockSize))

        if (class1Active && class1Level > 0.01) || (class2Active && class2Level > 0.01) {
            let sr = Float(sampleRate)
            var combSampleCount = 0
            for i in 0..<blockSize {
                let excIdx = (noiseIdx + i * noisePrimeStep) % noiseBuffer.count
                let exc = noiseBuffer[excIdx] * (0.25 + smoothedCentroid * 0.25) * 2.85
                var add: Float = 0
                if class1Active && class1Level > 0.01 && class1Id >= 0 {
                    let r = frequencyRange(for: class1Id)
                    add += class1Level * comb1.tick(exc * 0.5, delay: sr / (r.low + smoothedCentroid * (r.high - r.low))) * 0.7
                }
                if class2Active && class2Level > 0.01 && class2Id >= 0 {
                    let r = frequencyRange(for: class2Id)
                    add += class2Level * comb2.tick(exc * 0.5, delay: sr / (r.low + smoothedCentroid * (r.high - r.low))) * 0.7
                }
                ifftReal[i] = ifftReal[i] * 0.35 + add
            }
            // Debug: log comb activity periodically
            struct CombDebug { static var lastLog: Double = 0 }
            let now = CACurrentMediaTime()
            if now - CombDebug.lastLog > 1.0 {
                CombDebug.lastLog = now
                let r = frequencyRange(for: class1Id >= 0 ? class1Id : 0)
                print("[Audio] comb active: c1=\(class1Id)(\(String(format: "%.2f", class1Level))) c2=\(class2Id)(\(String(format: "%.2f", class2Level))) freq=\(String(format: "%.0f", r.low))-\(String(format: "%.0f", r.high))Hz centroid=\(String(format: "%.2f", smoothedCentroid))")
            }
        }

        if enableEdgeClicks && clickEnv > 0.001 {
            for i in 0..<min(16, blockSize) { ifftReal[i] += clickEnv * 0.15 * expf(-Float(i) / 4.0) }
            clickEnv *= 0.3
        } else { clickEnv *= 0.1 }

        var rms: Float = 0; vDSP_rmsqv(ifftReal, 1, &rms, vDSP_Length(blockSize))
        if rms > 1e-6 {
            var t = max(0.3, min(3.0, agcTargetRMS / rms))
            agcGain = agcAlpha * agcGain + (1 - agcAlpha) * t
            vDSP_vsmul(ifftReal, 1, &agcGain, &ifftReal, 1, vDSP_Length(blockSize))
        }
        var makeup = powf(10.0, outputGainDB / 20.0); vDSP_vsmul(ifftReal, 1, &makeup, &ifftReal, 1, vDSP_Length(blockSize))
        var peak: Float = 0; vDSP_maxmgv(ifftReal, 1, &peak, vDSP_Length(blockSize))
        if peak > 0.95 { var s = 0.95 / peak; vDSP_vsmul(ifftReal, 1, &s, &ifftReal, 1, vDSP_Length(blockSize)) }

        let theta = (-pan + 1) * Float.pi * 0.25; let gL = sin(theta), gR = cos(theta)
        for i in 0..<blockSize {
            let idx = (olaWrite + i) % olaL.count; olaL[idx] += ifftReal[i] * gL; olaR[idx] += ifftReal[i] * gR
        }
        olaWrite = (olaWrite + hop) % olaL.count
    }

    private func copyFromOLA(to outL: UnsafeMutablePointer<Float>, outR: UnsafeMutablePointer<Float>, count: Int) {
        for i in 0..<count {
            outL[i] = olaL[olaRead]; outR[i] = olaR[olaRead]
            olaL[olaRead] = 0; olaR[olaRead] = 0; olaRead = (olaRead + 1) % olaL.count
        }
    }

    private func applySpectralShaping() {
        bandLock.lock(); let edges = bandEdges; let b2b = binToBand; bandLock.unlock()
        guard edges.count == numBands + 1, !b2b.isEmpty else { return }
        let half = blockSize / 2; let binHz = Float(sampleRate / 2) / Float(half)
        for i in 0...half {
            guard i < b2b.count else { break }
            let band = max(0, min(b2b[i], numBands - 1))
            let f = Float(i) * binHz; let f0 = edges[band]; let f1 = edges[min(band+1, numBands)]
            let fc = sqrtf(max(10, f0) * max(10, f1)); let bw = max(60, f1 - f0)
            let tri = max(0, 1 - abs(f - fc) / (bw * 0.5))
            let w = min(1.0, max(0.05, currentGains[band])) * tri * (1.0 + (targetBoostLin - 1.0) * targetBoostBands[band])
            freqReal[i] *= w; freqImag[i] *= w
            if i > 0 && i < half { let m = blockSize - i; freqReal[m] *= w; freqImag[m] *= w }
        }
    }
}
