import Foundation
import ARKit
import UIKit
import Accelerate
import Combine
import Vision
import CoreML

final class DepthPipeline: NSObject, ObservableObject, ARSessionDelegate {
    static let gridWidth = 60
    static let gridHeight = 40

    @Published var debugImage: UIImage?
    @Published var fps: Double = 0
    @Published var scanColumn: Int = 0
    @Published var detectionStatusText: String = ""

    var nearMeters: Float = 0.30
    var farMeters: Float = 4.00
    var gainRangeDB: Float = 24

    private let session = ARSession()

    private var grid = [Float](repeating: 0, count: gridWidth * gridHeight)
    private let gridLock = NSLock()

    let detector = ObjectDetector()
    private var lastDetectionTime: TimeInterval = 0
    private let detectionInterval: TimeInterval = 0.20  // slightly slower to reduce frame pressure
    private var isDetectionRunning: Bool = false

    private var classGrid = [UInt8](repeating: 0, count: gridWidth * gridHeight)
    private var classGridVersion: UInt64 = 0

    private var activeClassIds: Set<Int> = []
    private let activeClassLock = NSLock()

    private var displayLink: CADisplayLink?
    private var sweepSeconds: Double = 2.0
    private var sweepStart = Date()

    private var onColumn: ((Int, [Float], [Float], Int, Float, Float, Float) -> Void)?

    private var lastTime = Date()
    private var frameCount = 0

    private var lastUIUpdateTime: TimeInterval = 0
    private let uiUpdateInterval: TimeInterval = 0.1
    private var lastDebugPrintTime: TimeInterval = 0
    private let debugPrintInterval: TimeInterval = 1.0

    func attach(to container: UIView) {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics.insert(.sceneDepth)
        config.environmentTexturing = .none
        session.delegate = self
        session.run(config)
    }

    func start(sweepSeconds: Double, onColumn: @escaping (Int, [Float], [Float], Int, Float, Float, Float) -> Void) {
        self.onColumn = onColumn
        self.sweepSeconds = sweepSeconds
        self.sweepStart = Date()

        displayLink?.invalidate()
        let dl = CADisplayLink(target: self, selector: #selector(step))
        dl.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
        dl.add(to: .main, forMode: .common)
        displayLink = dl
    }

    func stop() {
        displayLink?.invalidate(); displayLink = nil
        onColumn = nil
    }

    func setSweepRate(_ seconds: Double) {
        sweepSeconds = seconds
    }

    func setActiveClasses(_ ids: [Int]) {
        activeClassLock.lock()
        activeClassIds = Set(ids)
        activeClassLock.unlock()
        detector.setActiveClasses(ids)

        if ids.isEmpty {
            gridLock.lock()
            classGrid = [UInt8](repeating: 0, count: Self.gridWidth * Self.gridHeight)
            classGridVersion += 1
            gridLock.unlock()
            DispatchQueue.main.async { self.detectionStatusText = "" }
        }
    }

    @objc private func step() {
        let t = -sweepStart.timeIntervalSinceNow
        let period = max(0.2, sweepSeconds)
        let phase = (t.truncatingRemainder(dividingBy: period) + period)
                     .truncatingRemainder(dividingBy: period) / period

        let maxCol = Double(Self.gridWidth - 1)
        let col = Int(round(phase * maxCol))
        scanColumn = col

        let env = columnEnvelope(col: col)
        let (targetMask, classId) = columnTargetMaskAndClass(col: col)

        let norm = Double(col) / maxCol
        let pan = Float(norm * 2 - 1)

        var z01 = columnZ01(col: col)
        let edge01 = columnEdge01(col: col)

        let targetCov = columnTargetCoverage(col: col)
        z01 = clamp01(z01 - targetCov * 0.3)

        // Debug: log when we have a detected object in the sweep
        let maskSum = targetMask.reduce(0, +)
        if classId > 0 || maskSum > 0 {
            let now2 = CACurrentMediaTime()
            if now2 - lastDebugPrintTime > debugPrintInterval {
                lastDebugPrintTime = now2
                print("[Sweep] col=\(col) classId=\(classId) maskSum=\(maskSum) coverage=\(String(format: "%.2f", targetCov))")
            }
        }

        onColumn?(col, env, targetMask, classId, pan, z01, edge01)

        let now = CACurrentMediaTime()
        if now - lastUIUpdateTime > uiUpdateInterval {
            lastUIUpdateTime = now
            if let img = debugImageFromGrid(highlightCol: col) {
                DispatchQueue.main.async { self.debugImage = img }
            }
        }
    }

    private func columnTargetMaskAndClass(col: Int, bands: Int = DepthPipeline.gridHeight) -> ([Float], Int) {
        let W = Self.gridWidth
        let H = Self.gridHeight
        var mask = [Float](repeating: 0, count: bands)

        activeClassLock.lock()
        let targets = activeClassIds
        activeClassLock.unlock()

        guard !targets.isEmpty else { return (mask, -1) }

        gridLock.lock(); defer { gridLock.unlock() }

        var classCounts: [Int: Int] = [:]
        for y in 0..<H {
            let gridVal = Int(classGrid[y * W + col])
            if gridVal > 0 {
                let classId = gridVal - 1
                if targets.contains(classId) {
                    classCounts[classId, default: 0] += 1
                }
            }
        }

        guard !classCounts.isEmpty else { return (mask, -1) }

        let (bestClassId, _) = classCounts.max(by: { $0.value < $1.value })!

        for y in 0..<H {
            let gridVal = Int(classGrid[y * W + col])
            if gridVal > 0 && (gridVal - 1) == bestClassId {
                let rBand = (H - 1 - y)
                if rBand >= 0 && rBand < bands { mask[rBand] = 1.0 }
            }
        }

        return (mask, bestClassId)
    }

    // MARK: ARSessionDelegate

    private func handleFrame(_ frame: ARFrame) {
        guard let depthPB = frame.sceneDepth?.depthMap else { return }
        updateGrid(from: depthPB)

        let nowTime = frame.timestamp
        if nowTime - lastDetectionTime > detectionInterval && !isDetectionRunning {
            lastDetectionTime = nowTime
            // Pass capturedImage directly — do NOT hold a reference to frame
            let pixelBuffer = frame.capturedImage
            runDetection(pixelBuffer: pixelBuffer)
        }

        frameCount += 1
        let now = Date()
        let dt = now.timeIntervalSince(lastTime)
        if dt > 0.5 {
            fps = Double(frameCount) / dt
            frameCount = 0
            lastTime = now
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        handleFrame(frame)
    }

    func process(frame: ARFrame) {
        handleFrame(frame)
    }

    // MARK: - Detection

    private func runDetection(pixelBuffer: CVPixelBuffer) {
        guard !isDetectionRunning else { return }
        isDetectionRunning = true

        detector.detect(pixelBuffer: pixelBuffer, gridWidth: Self.gridWidth, gridHeight: Self.gridHeight) { [weak self] classGridResult, detections in
            guard let self else { return }
            defer { self.isDetectionRunning = false }

            if let newGrid = classGridResult {
                self.gridLock.lock()
                self.classGrid = newGrid
                self.classGridVersion += 1
                self.gridLock.unlock()
            } else {
                self.gridLock.lock()
                self.classGrid = [UInt8](repeating: 0, count: Self.gridWidth * Self.gridHeight)
                self.classGridVersion += 1
                self.gridLock.unlock()
            }

            let summary = detections?.map { "\($0.className) \(Int($0.confidence * 100))%" }.joined(separator: ", ") ?? ""
            DispatchQueue.main.async {
                self.detectionStatusText = summary
            }
        }
    }

    // MARK: - Depth Grid

    private func updateGrid(from pb: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }

        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb)?.assumingMemoryBound(to: Float32.self) else { return }

        gridLock.lock(); defer { gridLock.unlock() }

        for gy in 0..<Self.gridHeight {
            let y0 = gy * h / Self.gridHeight
            let y1 = min((gy + 1) * h / Self.gridHeight, h)
            for gx in 0..<Self.gridWidth {
                let x0 = gx * w / Self.gridWidth
                let x1 = min((gx + 1) * w / Self.gridWidth, w)
                var acc: Float = 0; var n = 0
                var yy = y0
                while yy < y1 {
                    var xx = x0; let row = yy * w
                    while xx < x1 { acc += base[row + xx]; n += 1; xx += 1 }
                    yy += 1
                }
                grid[gy * Self.gridWidth + gx] = n > 0 ? acc / Float(n) : 0
            }
        }
    }

    func columnEnvelope(col: Int) -> [Float] {
        gridLock.lock(); defer { gridLock.unlock() }
        var env = [Float](repeating: 1, count: Self.gridHeight)
        for r in 0..<Self.gridHeight {
            let gy = (Self.gridHeight - 1 - r)
            let d = grid[gy * Self.gridWidth + col]
            let t = clamp01(1 - (d - nearMeters) / max(0.001, (farMeters - nearMeters)))
            env[r] = pow(10.0, ((0.0 as Float) - gainRangeDB * (1 - t)) / 20.0)
        }
        smoothInPlace(&env, a: 0.6)
        return env
    }

    private func columnTargetCoverage(col: Int) -> Float {
        let W = Self.gridWidth, H = Self.gridHeight
        gridLock.lock(); defer { gridLock.unlock() }
        var count = 0
        for y in 0..<H { if classGrid[y * W + col] != 0 { count += 1 } }
        return Float(count) / Float(H)
    }

    private func columnZ01(col: Int) -> Float {
        let W = Self.gridWidth, H = Self.gridHeight
        gridLock.lock(); defer { gridLock.unlock() }
        var acc: Float = 0; let range = max(0.001, (farMeters - nearMeters))
        for y in 0..<H { acc += clamp01((grid[y * W + col] - nearMeters) / range) }
        return acc / Float(H)
    }

    private func columnEdge01(col: Int) -> Float {
        let W = Self.gridWidth, H = Self.gridHeight
        let c1 = min(W - 1, col + 1)
        gridLock.lock(); defer { gridLock.unlock() }
        var acc: Float = 0
        for y in 0..<H { acc += abs(grid[y * W + c1] - grid[y * W + col]) }
        let norm = (acc / Float(H)) / max(0.001, (farMeters - nearMeters))
        return clamp01(norm * 4)
    }

    private func smoothInPlace(_ x: inout [Float], a: Float) {
        guard x.count > 1 else { return }
        var prev = x[0]
        for i in 1..<x.count { prev = a * prev + (1 - a) * x[i]; x[i] = prev }
        prev = x.last ?? 0
        for i in (0..<(x.count-1)).reversed() { prev = a * prev + (1 - a) * x[i]; x[i] = prev }
    }

    private func debugImageFromGrid(highlightCol: Int) -> UIImage? {
        let W = Self.gridWidth, H = Self.gridHeight
        var rgba = [UInt8](repeating: 0, count: W * H * 4)

        gridLock.lock()
        let range = max(0.001, (farMeters - nearMeters))
        for y in 0..<H {
            for x in 0..<W {
                let idx = y * W + x
                let d = grid[idx]
                let t = clamp01(1 - (d - nearMeters) / range)
                let gray = UInt8(t * 255)
                let cls = classGrid[idx]
                let base = idx * 4

                if x == highlightCol {
                    rgba[base] = 255; rgba[base+1] = 0; rgba[base+2] = 0; rgba[base+3] = 255
                } else if cls == 0 {
                    rgba[base] = gray; rgba[base+1] = gray; rgba[base+2] = gray; rgba[base+3] = 255
                } else {
                    let classId = Int(cls) - 1
                    let hue = Float(classId * 37 % 360) / 360.0
                    let (r, g, b) = hsvToRgb(h: hue, s: 0.8, v: 1.0)
                    let oa: Float = 0.55; let gf = Float(gray)
                    rgba[base]   = UInt8(min(255, gf * (1-oa) + r * 255 * oa))
                    rgba[base+1] = UInt8(min(255, gf * (1-oa) + g * 255 * oa))
                    rgba[base+2] = UInt8(min(255, gf * (1-oa) + b * 255 * oa))
                    rgba[base+3] = 255
                }
            }
        }
        gridLock.unlock()

        let bytesPerRow = W * 4
        let cfData = CFDataCreate(nil, rgba, rgba.count)!
        let provider = CGDataProvider(data: cfData)!
        let cs = CGColorSpaceCreateDeviceRGB()
        let bi = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        if let cg = CGImage(width: W, height: H, bitsPerComponent: 8, bitsPerPixel: 32,
                            bytesPerRow: bytesPerRow, space: cs, bitmapInfo: bi,
                            provider: provider, decode: nil, shouldInterpolate: false,
                            intent: .defaultIntent) {
            return UIImage(cgImage: cg, scale: UIScreen.main.scale, orientation: .up)
        }
        return nil
    }

    private func hsvToRgb(h: Float, s: Float, v: Float) -> (Float, Float, Float) {
        let c = v * s; let x = c * (1 - abs(fmod(h * 6, 2) - 1)); let m = v - c
        let (r, g, b): (Float, Float, Float)
        let h6 = h * 6
        if h6 < 1      { (r,g,b) = (c,x,0) } else if h6 < 2 { (r,g,b) = (x,c,0) }
        else if h6 < 3 { (r,g,b) = (0,c,x) } else if h6 < 4 { (r,g,b) = (0,x,c) }
        else if h6 < 5 { (r,g,b) = (x,0,c) } else            { (r,g,b) = (c,0,x) }
        return (r+m, g+m, b+m)
    }
}

@inline(__always) private func clamp01(_ x: Float) -> Float { max(0, min(1, x)) }
