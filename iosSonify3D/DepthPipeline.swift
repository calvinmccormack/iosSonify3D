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
    @Published var isScanning: Bool = false
    @Published var detectionStatusText: String = ""

    var nearMeters: Float = 0.30
    var farMeters: Float = 4.00

    private let session = ARSession()

    // ─── Live grids (updated every AR frame) ───
    private var grid = [Float](repeating: 0, count: gridWidth * gridHeight)
    private var classGrid = [UInt8](repeating: 0, count: gridWidth * gridHeight)
    private let gridLock = NSLock()

    // ─── Snapshot grids (frozen when scan triggers) ───
    private var snapDepth = [Float](repeating: 0, count: gridWidth * gridHeight)
    private var snapClass = [UInt8](repeating: 0, count: gridWidth * gridHeight)

    let detector = ObjectDetector()
    private var lastDetectionTime: TimeInterval = 0
    private let detectionInterval: TimeInterval = 0.20
    private var isDetectionRunning: Bool = false

    private var activeClassIds: Set<Int> = []
    private var activeSlotMap: [Int: Int] = [:]   // classId → slot (0-3)
    private let activeClassLock = NSLock()

    // ─── Sweep state ───
    private var displayLink: CADisplayLink?
    private var sweepSeconds: Double = 2.0
    private var sweepStart = Date()

    /// Per-target hit: (slot, centroid, coverage, depth)
    struct ColumnHit {
        let slot: Int
        let centroid: Float
        let coverage: Float
    }
    private var onColumnMulti: ((Int, [ColumnHit], Float, Float) -> Void)?
    // Args: col, hits[], depth, pan

    private var lastTime = Date()
    private var frameCount = 0
    private var lastUIUpdateTime: TimeInterval = 0
    private let uiUpdateInterval: TimeInterval = 0.1

    func attach(to container: UIView) {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics.insert(.sceneDepth)
        config.environmentTexturing = .none
        session.delegate = self
        session.run(config)
    }

    func setSweepRate(_ seconds: Double) {
        sweepSeconds = seconds
    }

    func setActiveClasses(_ slotMap: [Int: Int]) {
        activeClassLock.lock()
        activeSlotMap = slotMap
        activeClassIds = Set(slotMap.keys)
        activeClassLock.unlock()
        detector.setActiveClasses(Array(slotMap.keys))

        if slotMap.isEmpty {
            gridLock.lock()
            classGrid = [UInt8](repeating: 0, count: Self.gridWidth * Self.gridHeight)
            gridLock.unlock()
            DispatchQueue.main.async { self.detectionStatusText = "" }
        }
    }

    // MARK: - Triggered Scan

    /// Snapshot current depth + class grids, then sweep once through the snapshot.
    /// onColumn args: col, [ColumnHit], depth, pan
    func triggerScan(sweepSeconds: Double,
                     onColumn: @escaping (Int, [ColumnHit], Float, Float) -> Void,
                     onComplete: @escaping () -> Void) {
        // Already scanning? ignore
        guard !isScanning else { return }

        self.sweepSeconds = sweepSeconds
        self.onColumnMulti = onColumn

        // Freeze snapshot
        gridLock.lock()
        snapDepth = grid
        snapClass = classGrid
        gridLock.unlock()

        print("[Scan] Snapshot taken. Sweep \(String(format: "%.1f", sweepSeconds))s")

        // Start sweep
        sweepStart = Date()
        DispatchQueue.main.async { self.isScanning = true }

        displayLink?.invalidate()
        let dl = CADisplayLink(target: SweepTarget(pipeline: self, onComplete: onComplete),
                               selector: #selector(SweepTarget.step))
        dl.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
        dl.add(to: .main, forMode: .common)
        displayLink = dl
    }

    func stopScan() {
        displayLink?.invalidate()
        displayLink = nil
        onColumnMulti = nil
        DispatchQueue.main.async { self.isScanning = false }
    }

    /// Helper class so CADisplayLink target doesn't retain DepthPipeline strongly in a cycle
    private class SweepTarget: NSObject {
        weak var pipeline: DepthPipeline?
        let onComplete: () -> Void

        init(pipeline: DepthPipeline, onComplete: @escaping () -> Void) {
            self.pipeline = pipeline
            self.onComplete = onComplete
        }

        @objc func step(_ link: CADisplayLink) {
            guard let p = pipeline else { link.invalidate(); return }
            let t = -p.sweepStart.timeIntervalSinceNow
            let period = max(0.2, p.sweepSeconds)

            if t >= period {
                // Sweep complete
                link.invalidate()
                p.displayLink = nil
                p.onColumnMulti = nil
                DispatchQueue.main.async {
                    p.isScanning = false
                    p.scanColumn = 0
                }
                print("[Scan] Complete")
                onComplete()
                return
            }

            let phase = t / period
            let maxCol = Double(DepthPipeline.gridWidth - 1)
            let col = Int(round(phase * maxCol))
            DispatchQueue.main.async { p.scanColumn = col }

            // Read from SNAPSHOT — get ALL target hits in this column
            let hits = p.snapColumnAllTargets(col: col)
            let z01 = p.snapColumnZ01(col: col)
            let pan = Float(Double(col) / maxCol * 2 - 1)

            // Debug: log hits periodically
            if !hits.isEmpty {
                struct HitDbg { static var lastLog: Double = 0 }
                let now2 = CACurrentMediaTime()
                if now2 - HitDbg.lastLog > 0.5 {
                    HitDbg.lastLog = now2
                    let desc = hits.map { "s\($0.slot) cov=\(String(format:"%.2f",$0.coverage)) cen=\(String(format:"%.2f",$0.centroid))" }.joined(separator: ", ")
                    print("[Sweep] col=\(col) hits: \(desc) z=\(String(format:"%.2f",z01))")
                }
            }

            p.onColumnMulti?(col, hits, z01, pan)

            // Update debug image periodically
            let now = CACurrentMediaTime()
            if now - p.lastUIUpdateTime > p.uiUpdateInterval {
                p.lastUIUpdateTime = now
                if let img = p.debugImageFromSnapshot(highlightCol: col) {
                    DispatchQueue.main.async { p.debugImage = img }
                }
            }
        }
    }

    // MARK: - Snapshot column queries

    /// Returns hits for ALL active target classes found in this column.
    private func snapColumnAllTargets(col: Int) -> [ColumnHit] {
        let W = Self.gridWidth, H = Self.gridHeight

        activeClassLock.lock()
        let slotMap = activeSlotMap
        let targets = activeClassIds
        activeClassLock.unlock()

        guard !targets.isEmpty else { return [] }

        // Count pixels per target class in this column
        var classCounts: [Int: Int] = [:]
        for y in 0..<H {
            let val = Int(snapClass[y * W + col])
            if val > 0 {
                let cid = val - 1
                if targets.contains(cid) { classCounts[cid, default: 0] += 1 }
            }
        }

        guard !classCounts.isEmpty else { return [] }

        var hits: [ColumnHit] = []
        for (classId, count) in classCounts {
            guard let slot = slotMap[classId] else { continue }
            let coverage = Float(count) / Float(H)

            // Compute Y centroid for this class in column
            var ws: Float = 0, tw: Float = 0
            for y in 0..<H {
                let val = Int(snapClass[y * W + col])
                if val > 0 && (val - 1) == classId {
                    let normY = Float(H - 1 - y) / Float(H - 1)
                    ws += normY; tw += 1
                }
            }
            let centroid = tw > 0 ? ws / tw : 0.5

            hits.append(ColumnHit(slot: slot, centroid: centroid, coverage: coverage))
        }

        return hits
    }

    /// Average normalized depth for this column from snapshot.
    private func snapColumnZ01(col: Int) -> Float {
        let W = Self.gridWidth, H = Self.gridHeight
        let range = max(0.001, farMeters - nearMeters)
        var acc: Float = 0
        for y in 0..<H {
            let d = snapDepth[y * W + col]
            acc += max(0, min(1, (d - nearMeters) / range))
        }
        return acc / Float(H)
    }

    // MARK: - ARSessionDelegate (runs continuously for live preview + YOLO)

    private func handleFrame(_ frame: ARFrame) {
        guard let depthPB = frame.sceneDepth?.depthMap else { return }
        updateGrid(from: depthPB)

        let nowTime = frame.timestamp
        if nowTime - lastDetectionTime > detectionInterval && !isDetectionRunning {
            lastDetectionTime = nowTime
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

        // Update debug image from live data when NOT scanning
        if !isScanning {
            let nowCA = CACurrentMediaTime()
            if nowCA - lastUIUpdateTime > uiUpdateInterval {
                lastUIUpdateTime = nowCA
                if let img = debugImageFromLive(highlightCol: nil) {
                    DispatchQueue.main.async { self.debugImage = img }
                }
            }
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

        detector.detect(pixelBuffer: pixelBuffer, gridWidth: Self.gridWidth,
                        gridHeight: Self.gridHeight) { [weak self] classGridResult, detections in
            guard let self else { return }
            defer { self.isDetectionRunning = false }

            if let newGrid = classGridResult {
                self.gridLock.lock()
                self.classGrid = newGrid
                self.gridLock.unlock()
            } else {
                self.gridLock.lock()
                self.classGrid = [UInt8](repeating: 0, count: Self.gridWidth * Self.gridHeight)
                self.gridLock.unlock()
            }

            let summary = detections?.map { "\($0.className) \(Int($0.confidence * 100))%" }
                .joined(separator: ", ") ?? ""
            DispatchQueue.main.async { self.detectionStatusText = summary }
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

    // MARK: - Debug images

    private func debugImageFromSnapshot(highlightCol: Int) -> UIImage? {
        return renderDebugImage(depthData: snapDepth, classData: snapClass, highlightCol: highlightCol)
    }

    private func debugImageFromLive(highlightCol: Int?) -> UIImage? {
        gridLock.lock()
        let d = grid
        let c = classGrid
        gridLock.unlock()
        return renderDebugImage(depthData: d, classData: c, highlightCol: highlightCol)
    }

    private func renderDebugImage(depthData: [Float], classData: [UInt8], highlightCol: Int?) -> UIImage? {
        let W = Self.gridWidth, H = Self.gridHeight
        var rgba = [UInt8](repeating: 0, count: W * H * 4)
        let range = max(0.001, farMeters - nearMeters)

        for y in 0..<H {
            for x in 0..<W {
                let idx = y * W + x
                let d = depthData[idx]
                let t = max(0, min(1, 1 - (d - nearMeters) / range))
                let gray = UInt8(t * 255)
                let cls = classData[idx]
                let base = idx * 4

                if let hc = highlightCol, x == hc {
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
